//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;
int THREAD_COUNT = 1;

PYBIND11_MODULE(raisim_env, m) {
  py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, RSG_MAKE_STR(RaisimWrapper))
      .def(py::init<std::string, std::string>(), py::arg("resourceDir"),
           py::arg("cfg"))
      .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
      .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
      .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
      .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
      .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
      .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
      .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
      .def("turnOnVisualization",
           &VectorizedEnvironment<ENVIRONMENT>::turnOnVisualization)
      .def("turnOffVisualization",
           &VectorizedEnvironment<ENVIRONMENT>::turnOffVisualization)
      .def("conditionalReset",
           &VectorizedEnvironment<ENVIRONMENT>::conditionalReset)
      .def("conditionalResetFlags",
           &VectorizedEnvironment<ENVIRONMENT>::getConditionalResetFlags)
      .def("getBasePosition",
           &VectorizedEnvironment<ENVIRONMENT>::getBasePosition)
      .def("getBaseOrientation",
           &VectorizedEnvironment<ENVIRONMENT>::getBaseOrientation)
      .def("setGoal", &VectorizedEnvironment<ENVIRONMENT>::setGoal)
      .def("getNominalJointPositions",
           &VectorizedEnvironment<ENVIRONMENT>::getNominalJointPositions)
      .def(py::pickle(
          [](const VectorizedEnvironment<ENVIRONMENT>&
                 p) {  // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
          },
          [](const py::tuple& t) {  // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<ENVIRONMENT> p(t[0].cast<std::string>(),
                                                 t[1].cast<std::string>());

            return p;
          }));

  py::class_<NormalSampler>(m, "NormalSampler")
      .def(py::init<int>(), py::arg("dim"))
      .def("seed", &NormalSampler::seed)
      .def("sample", &NormalSampler::sample);
}
