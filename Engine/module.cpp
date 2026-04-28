#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <memory>

#include "MLP.cpp" 

namespace py = pybind11;

PYBIND11_MODULE(CppModule, m) {
    m.doc() = "My Auto Grad Implementation with Memory-Safe Smart Pointers";

    // Bind Value
    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        // Constructors
        .def(py::init<double, std::string, std::vector<std::shared_ptr<Value>>, std::string>(), 
             py::arg("data"), py::arg("op") = "", py::arg("prev") = std::vector<std::shared_ptr<Value>>{}, py::arg("label") = "")
        .def(py::init<double>(), py::arg("data"))
        
        // Properties
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def_readwrite("label", &Value::label)
        .def_readwrite("_op", &Value::_op)
        .def_property_readonly("_prev", [](const std::shared_ptr<Value>& v) { return v->_prev; })
        
        // Methods
        .def("Tanh", &Value::Tanh)
        .def("Relu", &Value::Relu)
        .def("exp", &Value::exp)
        .def("log", &Value::log)
        .def("backward", &Value::backward)
        
        // Python Dunder Methods mapped to the global Operator Overloads in C++
        .def("__add__", [](std::shared_ptr<Value> a, std::shared_ptr<Value> b) { return a + b; })
        .def("__add__", [](std::shared_ptr<Value> a, double b) { return a + b; })
        .def("__radd__", [](std::shared_ptr<Value> a, double b) { return b + a; })
        
        .def("__mul__", [](std::shared_ptr<Value> a, std::shared_ptr<Value> b) { return a * b; })
        .def("__mul__", [](std::shared_ptr<Value> a, double b) { return a * b; })
        .def("__rmul__", [](std::shared_ptr<Value> a, double b) { return b * a; })
        
        .def("__sub__", [](std::shared_ptr<Value> a, std::shared_ptr<Value> b) { return a - b; })
        .def("__sub__", [](std::shared_ptr<Value> a, double b) { return a - b; })
        .def("__rsub__", [](std::shared_ptr<Value> a, double b) { return b - a; })
        
        .def("__neg__", [](std::shared_ptr<Value> a) { return a * -1.0; })
        
        .def("__truediv__", [](std::shared_ptr<Value> a, std::shared_ptr<Value> b) { return a / b; })
        .def("__truediv__", [](std::shared_ptr<Value> a, double b) { return a / b; })
        .def("__rtruediv__", [](std::shared_ptr<Value> a, double b) { return b / a; })
        
        .def("__pow__", [](std::shared_ptr<Value> a, double b) { return pow(a, b); })
        
        .def("__repr__", [](const std::shared_ptr<Value>& v) {
            std::ostringstream os;
            os << "Value(data=" << v->data << ", grad=" << v->grad << ", label='" << v->label << "')";
            return os.str();
        });

    // Bind Neuron
    py::class_<Neuron>(m, "Neuron")
        .def(py::init<int, std::string>(), py::arg("in_dim"), py::arg("nonlin") = "")
        .def_readwrite("nonlin", &Neuron::nonlin)
        .def_readwrite("W", &Neuron::W)
        .def_readwrite("b", &Neuron::b)
        .def("__call__", py::overload_cast<const std::vector<double>&>(&Neuron::operator()))
        .def("__call__", py::overload_cast<const std::vector<std::shared_ptr<Value>>&>(&Neuron::operator()))
        .def("parameters", &Neuron::parameters);

    // Bind Layer
    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int, std::string>(), py::arg("in_dim"), py::arg("out_dim"), py::arg("nonlin") = "")
        .def_readwrite("neurons", &Layer::neurons)
        
        // FIX 1: Changed MLP& self to Layer& self
        .def("__call__", [](Layer& self, const std::vector<double>& X) -> py::object {
            auto outs = self(X);
            if (outs.size() == 1) {
                return py::cast(outs[0]);
            }
            return py::cast(outs);
        })
        
        // FIX 2: Applied the same dynamic return lambda to the shared_ptr overload
        .def("__call__", [](Layer& self, const std::vector<std::shared_ptr<Value>>& X) -> py::object {
            auto outs = self(X);
            if (outs.size() == 1) {
                return py::cast(outs[0]);
            }
            return py::cast(outs);
        })
        
        .def("parameters", &Layer::parameters);

    // Bind MLP
    py::class_<MLP>(m, "MLP")
        .def(py::init<>())
        .def_readwrite("layers", &MLP::layers)
        .def("Linear", &MLP::Linear, py::arg("in_dim"), py::arg("out_dim"))
        .def("Tanh", &MLP::Tanh)
        .def("Relu", &MLP::Relu)
        .def("__call__", [](MLP& self, const std::vector<double>& X) -> py::object {
            auto outs = self(X);
            if (outs.size() == 1) {
                return py::cast(outs[0]);
            }
            return py::cast(outs);
        })
        .def("parameters", &MLP::parameters)
        .def("zero_grad", &MLP::zero_grad)
        .def("step", &MLP::step, py::arg("lr") = 0.01)
        .def("__repr__", &MLP::repr);
}