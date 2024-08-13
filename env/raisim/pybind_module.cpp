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
  py::class_<VectorizedEnvironment<Environment>>(m, RSG_MAKE_STR(RaisimWrapper))
      .def(py::init<std::string, std::string>(), py::arg("resourceDir"),
           py::arg("cfg"))
      .def("reset", &VectorizedEnvironment<Environment>::reset)
      .def("observe", &VectorizedEnvironment<Environment>::observe)
      .def("step", &VectorizedEnvironment<Environment>::step)
      .def("setSeed", &VectorizedEnvironment<Environment>::setSeed)
      .def("getObDim", &VectorizedEnvironment<Environment>::getObDim)
      .def("getActionDim", &VectorizedEnvironment<Environment>::getActionDim)
      .def("getNumOfEnvs", &VectorizedEnvironment<Environment>::getNumOfEnvs)
      .def("turnOnVisualization",
           &VectorizedEnvironment<Environment>::turnOnVisualization)
      .def("turnOffVisualization",
           &VectorizedEnvironment<Environment>::turnOffVisualization)
      .def("conditionalReset",
           &VectorizedEnvironment<Environment>::conditionalReset)
      .def("conditionalResetFlags",
           &VectorizedEnvironment<Environment>::getConditionalResetFlags)
      .def("getBasePosition",
           &VectorizedEnvironment<Environment>::getBasePosition)
      .def("getBaseOrientation",
           &VectorizedEnvironment<Environment>::getBaseOrientation)
      .def("setGoal", &VectorizedEnvironment<Environment>::setGoal)
      .def("getNominalJointPositions",
           &VectorizedEnvironment<Environment>::getNominalJointPositions)
      .def(py::pickle(
          [](const VectorizedEnvironment<Environment>&
                 p) {  // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
          },
          [](const py::tuple& t) {  // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<Environment> p(t[0].cast<std::string>(),
                                                 t[1].cast<std::string>());

            return p;
          }));

  py::class_<NormalSampler>(m, "NormalSampler")
      .def(py::init<int>(), py::arg("dim"))
      .def("seed", &NormalSampler::seed)
      .def("sample", &NormalSampler::sample);
}
