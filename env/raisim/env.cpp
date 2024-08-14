//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <cstdlib>
#include <set>

#include "yaml/Yaml.hpp"
#include "actuation_dynamics/Actuation.hpp"
#include "command.cpp"
#include "observation.cpp"
#include "visualization.cpp"

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
    /// create world
    world_ = std::make_unique<raisim::World>();

    setControlTimeStep(cfg["control_dt"].template As<double>());
    setSimulationTimeStep(cfg["simulation_dt"].template As<double>());

    /// add objects
    robot_ = world_->addArticulatedSystem(
        resourceDir_ + "/models/anymal_d/urdf/anymal_d.urdf");
    robot_->setName("anymal_c");
    robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    baseMassMean_ = robot_->getMass(0);

    auto ground = world_->addGround(0., "ground_material");

    /// get robot data
    gcDim_ = static_cast<int>(robot_->getGeneralizedCoordinateDim());
    gvDim_ = static_cast<int>(robot_->getDOF());
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);
    prevPTarget12_.setZero(nJoints_);
    contacts_.setZero(4);

    /// this is nominal configuration of anymal_c
    gc_init_ = observationHandler_.getNominalGeneralizedCoordinates();

    useActuatorNetwork_ = cfg["use_actuator_network"].template As<bool>();

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointDgain.setZero();

    if (!useActuatorNetwork_) {
      jointPgain.tail(nJoints_).setConstant(80.0);
      jointDgain.tail(nJoints_).setConstant(2.0);
    }

    robot_->setPdGains(jointPgain, jointDgain);
    robot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 48;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    prevPTarget12_ = actionMean_;

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(cfg["action_scaling"].template As<double>());

    /// Set the material property for each of the collision bodies of the robot
    for (auto &collisionBody : robot_->getCollisionBodies()) {
      if (collisionBody.colObj->name.find("FOOT") != std::string::npos) {
        collisionBody.setMaterial("foot_material");
      } else {
        collisionBody.setMaterial("robot_material");
      }
    }

    auto materialPairGroundFootProperties =
        world_->getMaterialPairProperties("ground_material", "foot_material");
    world_->setMaterialPairProp("ground_material", "foot_material", 0.6,
                                materialPairGroundFootProperties.c_r,
                                materialPairGroundFootProperties.r_th);

    auto materialPairGroundRobotProperties =
        world_->getMaterialPairProperties("ground_material", "robot_material");
    world_->setMaterialPairProp("ground_material", "robot_material", 0.4,
                                materialPairGroundRobotProperties.c_r,
                                materialPairGroundRobotProperties.r_th);

    /// Episode Length
    maxEpisodeLength_ = std::floor(cfg_["max_time"].template As<double>() /
                                   cfg_["control_dt"].template As<double>());

    /// Dynamics Randomization
    enableDynamicsRandomization_ =
        cfg["enable_dynamics_randomization"].template As<bool>();

    /// Frame Indices
    frameIdxs_.setZero(5);
    frameIdxs_ << robot_->getFrameIdxByName("ROOT"),
        robot_->getFrameIdxByName("LF_shank_fixed_LF_FOOT"),
        robot_->getFrameIdxByName("RF_shank_fixed_RF_FOOT"),
        robot_->getFrameIdxByName("LH_shank_fixed_LH_FOOT"),
        robot_->getFrameIdxByName("RH_shank_fixed_RH_FOOT");

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(cfg["server_port"].template As<int>());
      server_->focusOn(robot_);
      visualizationHandler_.setServer(server_);
      server_->addVisualSphere("goal", 0.1, 1, 0, 0, 1);
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
        actuationPositionErrorInputScaling_ =
            std::clamp(1. + normalDistribution_(gen_) * 0.05, 0.975, 1.025);
        actuationVelocityInputScaling_ =
            std::clamp(1. + normalDistribution_(gen_) * 0.025, 0.975, 1.025);
        actuationOutputTorqueScaling_ =
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
    prevPTarget12_ = actionMean_;

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

    float jointTorqueSquaredNorm = 0.f;

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();

      /// Use the actuation network to compute torques
      if (i % int(0.005 / simulation_dt_ + 1e-10) == 0 && useActuatorNetwork_) {
        robot_->getState(gc_, gv_);

        Eigen::VectorXd gf(gvDim_);
        gf.setZero();

        gf.tail(12) = actuationOutputTorqueScaling_ *
                      actuation_
                          .getActuationTorques(
                              actuationPositionErrorInputScaling_ *
                                  (pTarget12_ - gc_.tail(12)),
                              actuationVelocityInputScaling_ * gv_.tail(12))
                          .cwiseMax(-80.)
                          .cwiseMin(80.);

        robot_->setGeneralizedForce(gf);
      } else if (!useActuatorNetwork_) {
        robot_->setPdTarget(pTarget_, Eigen::VectorXd::Zero(gvDim_));
      }

      world_->integrate();

      jointTorqueSquaredNorm = robot_->getGeneralizedForce().squaredNorm();

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
      if ((contact.getCollisionBodyA()->material == "ground_material" &&
           contact.getCollisionBodyB()->material != "foot_material") ||
          (contact.getCollisionBodyA()->material == "foot_material" &&
           contact.getCollisionBodyB()->material != "ground_material")) {
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

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_->setTimeStep(dt);
  }

  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return obDim_; }

  int getActionDim() { return actionDim_; }

 private:
  std::unique_ptr<raisim::World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::string resourceDir_;
  Yaml::Node cfg_;
  int obDim_ = 0, actionDim_ = 0;
  std::unique_ptr<raisim::RaisimServer> server_;

  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false, visualizing_ = false;
  raisim::ArticulatedSystem *robot_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_,
      prevPTarget12_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, contacts_, frameIdxs_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, goalPosition_;

  // Actuator network
  Actuation actuation_;

  // Conditional Reset
  int maxEpisodeLength_;
  int stepCount_ = 0;

  /// Dynamics Randomization Parameters
  bool enableDynamicsRandomization_ = false;
  double baseMassMean_;

  bool useActuatorNetwork_ = true;

  double actuationPositionErrorInputScaling_ = 1.;
  double actuationVelocityInputScaling_ = 1.;
  double actuationOutputTorqueScaling_ = 1.;

  // Randomization engine
  std::normal_distribution<double> normalDistribution_;
  std::uniform_real_distribution<double> uniformRealDistribution_;
  thread_local static std::mt19937 gen_;

  // Helper Classes
  VelocityCommand velocityCommandHandler_;
  ObservationHandler observationHandler_;
  VisualizationHandler visualizationHandler_;
};

thread_local std::mt19937 raisim::Env::gen_;
}  // namespace raisim