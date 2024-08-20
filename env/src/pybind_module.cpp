//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "env.cpp"
#include "vec_env.cpp"

namespace py = pybind11;
using namespace raisim;
int THREAD_COUNT = 1;

PYBIND11_MODULE(raisim_env, m) {
  py::class_<VecEnv<Env>>(m, RSG_MAKE_STR(RaisimWrapper))
      .def(py::init<std::string, std::string>(), py::arg("resourceDir"),
           py::arg("cfg"))
      .def("reset", &VecEnv<Env>::reset)
      .def("observe", &VecEnv<Env>::observe)
      .def("step", &VecEnv<Env>::step)
      .def("setSeed", &VecEnv<Env>::setSeed)
      .def("getObDim", &VecEnv<Env>::getObDim)
      .def("getActionDim", &VecEnv<Env>::getActionDim)
      .def("getNumOfEnvs", &VecEnv<Env>::getNumOfEnvs)
      .def("turnOnVisualization", &VecEnv<Env>::turnOnVisualization)
      .def("turnOffVisualization", &VecEnv<Env>::turnOffVisualization)
      .def("conditionalReset", &VecEnv<Env>::conditionalReset)
      .def("conditionalResetFlags", &VecEnv<Env>::getConditionalResetFlags)
      .def("getBasePosition", &VecEnv<Env>::getBasePosition)
      .def("getBaseOrientation", &VecEnv<Env>::getBaseOrientation)
      .def("setGoal", &VecEnv<Env>::setGoal)
      .def("getNominalJointPositions", &VecEnv<Env>::getNominalJointPositions)
      .def("getTorques", &VecEnv<Env>::getTorques)
      .def(py::pickle(
          [](const VecEnv<Env>& p) {  // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
          },
          [](const py::tuple& t) {  // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VecEnv<Env> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
          }));
}
