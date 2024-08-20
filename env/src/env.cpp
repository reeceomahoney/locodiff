//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <cstdlib>
#include <set>

#include "actuation_dynamics/Actuation.hpp"
#include "command.cpp"
#include "observation.cpp"
#include "visualization.cpp"
#include "yaml/Yaml.hpp"

using EigenVec = Eigen::Matrix<float, -1, 1>;

namespace raisim {

class Env {
 public:
  explicit Env(const std::string &resourceDir, const Yaml::Node &cfg,
               bool visualizable)
      : resourceDir_(std::move(resourceDir)),
        cfg_(cfg),
        visualizable_(visualizable),
        normalDistribution_(0, 1),
        uniformRealDistribution_(-1, 1),
        actuation_(resourceDir + "/parameters/anymal_c_actuation",
                   Eigen::Vector2d{1., 0.1}, 100., 12),
        velocityCommandHandler_(cfg["velocity_command"],
                                cfg["control_dt"].template As<double>()),
        visualizationHandler_(visualizable) {
    initWorld();
    initRobot();
    initContainers();
    initGains();
    initMaterials();
    initServer();
  }

  void initWorld() {
    world_ = std::make_unique<raisim::World>();
    setControlTimeStep(cfg_["control_dt"].template As<double>());
    setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
    maxEpisodeLength_ = std::floor(cfg_["max_time"].template As<double>() /
                                   cfg_["control_dt"].template As<double>());
    enableDynamicsRandomization_ =
        cfg_["enable_dynamics_randomization"].template As<bool>();
  }

  void initRobot() {
    robot_ = world_->addArticulatedSystem(
        resourceDir_ + "/models/anymal_d/urdf/anymal_d.urdf");
    robot_->setName("anymal_d");
    robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  }

  void initContainers() {
    obDim_ = 48;
    gcDim_ = static_cast<int>(robot_->getGeneralizedCoordinateDim());
    gvDim_ = static_cast<int>(robot_->getDOF());
    nJoints_ = gvDim_ - 6;
    baseMassMean_ = robot_->getMass(0);

    gc_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gc_init_.setZero(gcDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    pTarget12_.setZero(nJoints_);
    gf_.setZero(gvDim_);
    gc_init_ = observationHandler_.getNominalGeneralizedCoordinates();
  }

  void initGains() {
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    useActuatorNetwork_ = cfg_["use_actuator_network"].template As<bool>();
    jointPgain.setZero();
    jointDgain.setZero();

    if (!useActuatorNetwork_) {
      jointPgain.tail(nJoints_).setConstant(80.0);
      jointDgain.tail(nJoints_).setConstant(2.0);
    }

    robot_->setPdGains(jointPgain, jointDgain);
    robot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
  }

  void initMaterials() {
    world_->addGround(0., "ground");

    for (auto &collisionBody : robot_->getCollisionBodies()) {
      if (collisionBody.colObj->name.find("FOOT") != std::string::npos) {
        collisionBody.setMaterial("foot");
      } else {
        collisionBody.setMaterial("body");
      }
    }

    // set friction properties
    auto groundFootProps = world_->getMaterialPairProperties("ground", "foot");
    world_->setMaterialPairProp("ground", "foot", 0.6, groundFootProps.c_r,
                                groundFootProps.r_th);
    auto groundRobotProbs = world_->getMaterialPairProperties("ground", "body");
    world_->setMaterialPairProp("ground", "body", 0.4, groundRobotProbs.c_r,
                                groundRobotProbs.r_th);
  }

  void initServer() {
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(cfg_["server_port"].template As<int>());
      server_->focusOn(robot_);
      visualizationHandler_.setServer(server_);
    }
  }

  void reset() {
    Eigen::VectorXd gc = gc_init_, gv = gv_init_;

    if (enableDynamicsRandomization_) {
      gc[2] += 0.0 * std::abs(uniformRealDistribution_(gen_));

      gc.tail(12) += 0.1 * Eigen::VectorXd::NullaryExpr(12, [&]() {
                       return uniformRealDistribution_(gen_);
                     });

      gv.head(3) += 0.1 * Eigen::VectorXd::NullaryExpr(3, [&]() {
                      return uniformRealDistribution_(gen_);
                    });
      gv.segment(3, 3) += 0.15 * Eigen::VectorXd::NullaryExpr(3, [&]() {
                            return uniformRealDistribution_(gen_);
                          });
      gv.tail(12) += 0.05 * Eigen::VectorXd::NullaryExpr(12, [&]() {
                       return uniformRealDistribution_(gen_);
                     });
    }

    robot_->setState(gc, gv);

    if (enableDynamicsRandomization_) {
      robot_->setMass(
          0, baseMassMean_ +
                 std::clamp(normalDistribution_(gen_) * 10., -15., 15.));
      robot_->updateMassInfo();

      auto materialPairGroundFootProperties =
          world_->getMaterialPairProperties("ground_material", "foot_material");
      world_->setMaterialPairProp(
          "ground_material", "foot_material",
          std::clamp(0.6 + normalDistribution_(gen_) * 0.5, 0.1, 2.0),
          materialPairGroundFootProperties.c_r,
          materialPairGroundFootProperties.r_th);

      if (useActuatorNetwork_) {
        actPosErrScaling_ =
            std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.975, 1.025);
        actVelScaling_ =
            std::clamp(1. + normalDistribution_(gen_) * 0.025, 0.975, 1.025);
        actOutScaling_ =
            std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.95, 1.05);
      } else {
        Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);

        jointPgain.setZero();
        jointDgain.setZero();

        jointPgain.tail(nJoints_).setConstant(
            80.0 *
            (std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.975, 1.025)));
        jointDgain.tail(nJoints_).setConstant(
            2.0 *
            (std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.975, 1.025)));

        robot_->setPdGains(jointPgain, jointDgain);
      }
    }

    velocityCommandHandler_.reset(robot_->getBaseOrientation().e());
    observationHandler_.reset(robot_,
                              velocityCommandHandler_.getVelocityCommand());
    actuation_.reset();
    stepCount_ = 0;
  }

  bool conditionalReset() {
    if (stepCount_ >= maxEpisodeLength_) {
      reset();
      return true;
    }

    return false;
  }

  void step(const Eigen::Ref<EigenVec> &action) {
    pTarget12_ = action.cast<double>();
    pTarget_.tail(nJoints_) = pTarget12_;

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();

      /// Use the actuation network to compute torques
      if (i % int(0.005 / simulation_dt_ + 1e-10) == 0 && useActuatorNetwork_) {
        robot_->getState(gc_, gv_);
        gf_.setZero();

        auto actPosErr = actPosErrScaling_ * (pTarget12_ - gc_.tail(12));
        auto actVel = actVelScaling_ * gv_.tail(12);
        gf_.tail(12) =
            actOutScaling_ * actuation_.getActuationTorques(actPosErr, actVel);
        gf_ = gf_.cwiseMax(-80.).cwiseMin(80.);

        robot_->setGeneralizedForce(gf_);
      } else if (!useActuatorNetwork_) {
        robot_->setPdTarget(pTarget_, Eigen::VectorXd::Zero(gvDim_));
      }

      world_->integrate();

      observationHandler_.updateObservation(robot_, pTarget12_);

      if (visualizing_ && visualizable_) {
        visualizationHandler_.updateVelocityVisual(robot_, observationHandler_,
                                                   server_);
      }

      if (server_) server_->unlockVisualizationServerMutex();
    }

    velocityCommandHandler_.step(robot_->getBaseOrientation().e());
    observationHandler_.updateVelocityCommand(
        velocityCommandHandler_.getVelocityCommand());
    observationHandler_.updateJointHistory(pTarget12_);

    ++stepCount_;
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    /// convert it to float
    ob = observationHandler_.getObservation().cast<float>();
  }

  bool isTerminalState() {
    /// if the contact body is not feet
    for (auto &contact : robot_->getContacts()) {
      if ((contact.getCollisionBodyA()->material == "ground" &&
           contact.getCollisionBodyB()->material != "foot") ||
          (contact.getCollisionBodyA()->material == "foot" &&
           contact.getCollisionBodyB()->material != "ground")) {
        return true;
      }
    }

    if (stepCount_ >= maxEpisodeLength_) {
      return true;
    }

    return false;
  }

  void setSeed(int seed) {
    gen_.seed(seed);
    srand(seed);

    velocityCommandHandler_.setSeed(seed);
  }

  void turnOnVisualization() {
    server_->wakeup();
    visualizing_ = true;
  }

  void turnOffVisualization() {
    server_->hibernate();
    visualizing_ = false;
  }

  void getBasePosition(Eigen::Ref<EigenVec> basePosition) {
    basePosition = robot_->getBasePosition().e().cast<float>();
  }

  void getBaseOrientation(Eigen::Ref<EigenVec> baseOrientation) {
    baseOrientation = observationHandler_.getGeneralizedCoordinate()
                          .segment(3, 4)
                          .cast<float>();
  }

  void setGoal(const Eigen::Ref<EigenVec> &goal) {
    goalPosition_ << goal[0], goal[1], 0;
    server_->getVisualObject("goal")->setPosition(goalPosition_);
  }

  void getNominalJointPositions(Eigen::Ref<EigenVec> nominalJointPositions) {
    nominalJointPositions =
        observationHandler_.getNominalGeneralizedCoordinates()
            .tail(nJoints_)
            .cast<float>();
  }

  void getTorques(Eigen::Ref<EigenVec> torques) {
    torques = gf_.tail(nJoints_).cast<float>();
  }

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_->setTimeStep(dt);
  }

  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return obDim_; }

  int getActionDim() { return nJoints_; }

 private:
  // world and simulation
  std::unique_ptr<raisim::World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::string resourceDir_;
  Yaml::Node cfg_;

  // dimensions
  int obDim_ = 0;
  int gcDim_, gvDim_, nJoints_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, gf_;
  Eigen::Vector3d goalPosition_;

  // robot and server
  raisim::ArticulatedSystem *robot_;
  std::unique_ptr<raisim::RaisimServer> server_;
  bool visualizable_ = false, visualizing_ = false;

  // Actuation
  Actuation actuation_;
  double actPosErrScaling_ = 1.;
  double actVelScaling_ = 1.;
  double actOutScaling_ = 1.;
  bool useActuatorNetwork_ = true;

  // episode
  int maxEpisodeLength_;
  int stepCount_ = 0;

  // dynamics dandomization
  bool enableDynamicsRandomization_ = false;
  double baseMassMean_;

  // randomization engine
  std::normal_distribution<double> normalDistribution_;
  std::uniform_real_distribution<double> uniformRealDistribution_;
  thread_local static std::mt19937 gen_;

  // helpers
  VelocityCommand velocityCommandHandler_;
  ObservationHandler observationHandler_;
  VisualizationHandler visualizationHandler_;
};

thread_local std::mt19937 raisim::Env::gen_;
}  // namespace raisim