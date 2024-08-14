//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include <Eigen/Core>

#include "Yaml.hpp"
#include "omp.h"

using EigenRowMajorMat = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
using EigenVec = Eigen::Matrix<float, -1, 1>;
using EigenBoolVec = Eigen::Matrix<bool, -1, 1>;

namespace raisim {

int THREAD_COUNT;

template <class ChildEnvironment>
class VecEnv {
 public:
  explicit VecEnv(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir), cfgString_(cfg) {
    Yaml::Parse(cfg_, cfg);
    if (&cfg_["render"]) render_ = cfg_["render"].template As<bool>();
    init();
    earlyTerminationActive_ = cfg_["early_termination"].template As<bool>();
    normalizeObservation_ = cfg_["normalize_observation"].template As<bool>();
  }

  ~VecEnv() {
    for (auto *ptr : environments_) delete ptr;
  }

  const std::string &getResourceDir() const { return resourceDir_; }

  const std::string &getCfgString() const { return cfgString_; }

  const bool &getNormalizeObservation() const { return normalizeObservation_; }

  void init() {
    THREAD_COUNT = cfg_["num_threads"].template As<int>();
    omp_set_num_threads(THREAD_COUNT);
    num_envs_ = cfg_["num_envs"].template As<int>();

    if (!environments_.empty()) {
      for (auto *ptr : environments_) delete ptr;

      environments_.clear();
    }

    if (!conditionalResetPerformed_.empty()) {
      conditionalResetPerformed_.clear();
    }

    environments_.reserve(num_envs_);
    conditionalResetPerformed_.reserve(num_envs_);

    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(
          new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(
          cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(
          cfg_["control_dt"].template As<double>());
      conditionalResetPerformed_.push_back(true);
    }

    setSeed(0);

    for (int i = 0; i < num_envs_; i++) {
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0,
               "Observation/Action dimension must be defined in the "
               "constructor of each environment!")

    /// ob scaling
    if (normalizeObservation_) {
      obMean_.setZero(obDim_);
      obVar_.setOnes(obDim_);
      recentMean_.setZero(obDim_);
      recentVar_.setZero(obDim_);
      delta_.setZero(obDim_);
      epsilon.setZero(obDim_);
      epsilon.setConstant(1e-8);
    }
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env : environments_) env->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++) environments_[i]->observe(ob.row(i));

    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(ob, updateStatistics);
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++) perAgentStep(i, action, done);
  }

  void turnOnVisualization() {
    if (render_) environments_[0]->turnOnVisualization();
  }

  void turnOffVisualization() {
    if (render_) environments_[0]->turnOffVisualization();
  }

  void setSeed(int seed) {
    int seed_inc = num_envs_ * seed;
    for (auto *env : environments_) env->setSeed(seed_inc++);
  }

  int getObDim() { return obDim_; }

  int getActionDim() { return actionDim_; }

  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void conditionalReset() {
    for (int i = 0; i < num_envs_; i++) {
      conditionalResetPerformed_[i] = environments_[i]->conditionalReset();
    }
  };

  const std::vector<bool> &getConditionalResetFlags() {
    return conditionalResetPerformed_;
  }

  void getBasePosition(Eigen::Ref<EigenRowMajorMat> &position) {
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->getBasePosition(position.row(i));
  }

  void getBaseOrientation(Eigen::Ref<EigenRowMajorMat> &orientation) {
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->getBaseOrientation(orientation.row(i));
  }

  void setGoal(Eigen::Ref<EigenRowMajorMat> &goal) {
    environments_[0]->setGoal(goal.row(0));
  }

  void getNominalJointPositions(Eigen::Ref<EigenRowMajorMat> &nominalJointPos) {
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->getNominalJointPositions(nominalJointPos.row(i));
  }

 private:
  void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob,
                                               bool updateStatistics) {
    if (updateStatistics) {
      recentMean_ = ob.colwise().mean();
      recentVar_ =
          (ob.rowwise() - recentMean_.transpose()).colwise().squaredNorm() /
          num_envs_;

      delta_ = obMean_ - recentMean_;
      for (int i = 0; i < obDim_; i++) delta_[i] = delta_[i] * delta_[i];

      float totCount = obCount_ + num_envs_;

      obMean_ = obMean_ * (obCount_ / totCount) +
                recentMean_ * (num_envs_ / totCount);
      obVar_ = (obVar_ * obCount_ + recentVar_ * num_envs_ +
                delta_ * (obCount_ * num_envs_ / totCount)) /
               (totCount);
      obCount_ = totCount;
    }

#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      ob.row(i) = (ob.row(i) - obMean_.transpose())
                      .template cwiseQuotient<>(
                          (obVar_ + epsilon).cwiseSqrt().transpose());
  }

  inline void perAgentStep(int agentId, Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenBoolVec> &done) {
    environments_[agentId]->step(action.row(agentId));

    done[agentId] = environments_[agentId]->isTerminalState();

    if (done[agentId] && earlyTerminationActive_) {
      environments_[agentId]->reset();
    } else {
      done[agentId] = false;
    }
  }

  std::vector<ChildEnvironment *> environments_;

  std::vector<bool> conditionalResetPerformed_;

  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  bool recordVideo_ = false, render_ = false;
  bool earlyTerminationActive_ = true;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;

  /// observation running mean
  bool normalizeObservation_ = true;
  EigenVec obMean_;
  EigenVec obVar_;
  float obCount_ = 1e-4;
  EigenVec recentMean_, recentVar_, delta_;
  EigenVec epsilon;
};

}  // namespace raisim

#endif  // SRC_RAISIMGYMVECENV_HPP
