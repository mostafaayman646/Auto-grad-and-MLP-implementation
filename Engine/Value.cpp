#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <memory>
#include <unordered_set>

using namespace std;

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    string _op;
    vector<shared_ptr<Value>> _prev;
    string label;
    function<void()> _backward;

    Value(double data, string op = "", vector<shared_ptr<Value>> prev = {}, string label = "") 
        : data(data), grad(0.0), _op(op), _prev(prev), label(label), _backward([](){}) {}

    shared_ptr<Value> Tanh() {
        double t = (exp(2 * data) - 1) / (exp(2 * data) + 1);
        auto out = make_shared<Value>(t, "Tanh", vector<shared_ptr<Value>>{shared_from_this()});
        
        out->_backward = [this, out]() {
            this->grad += (1 - std::pow(out->data, 2)) * out->grad;
        };
        return out;
    }

    shared_ptr<Value> Relu() {
        double relu_val = data > 0 ? data : 0.0;
        auto out = make_shared<Value>(relu_val, "Relu", vector<shared_ptr<Value>>{shared_from_this()});
        
        out->_backward = [this, out]() {
            this->grad += (out->data > 0 ? 1.0 : 0.0) * out->grad;
        };
        return out;
    }

    void backward() {
        vector<shared_ptr<Value>> topo;
        unordered_set<Value*> visited;
        
        // Topological Sort
        function<void(shared_ptr<Value>)> build_topo = [&](shared_ptr<Value> v) {
            if (visited.find(v.get()) == visited.end()) {
                visited.insert(v.get());
                for (auto child : v->_prev) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };

        build_topo(shared_from_this());

        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

    friend ostream& operator<<(ostream& os, const shared_ptr<Value>& v) {
        os << "Value(data=" << v->data << ", grad=" << v->grad << ")";
        return os;
    }
};

inline shared_ptr<Value> operator+(shared_ptr<Value> lhs, shared_ptr<Value> rhs) {
    auto out = make_shared<Value>(lhs->data + rhs->data, "+", vector<shared_ptr<Value>>{lhs, rhs});
    out->_backward = [lhs, rhs, out]() {
        lhs->grad += out->grad;
        rhs->grad += out->grad;
    };
    return out;
}

inline shared_ptr<Value> operator+(shared_ptr<Value> lhs, double rhs) {
    return lhs + make_shared<Value>(rhs);
}

inline shared_ptr<Value> operator+(double lhs, shared_ptr<Value> rhs) {
    return make_shared<Value>(lhs) + rhs;
}

inline shared_ptr<Value> operator*(shared_ptr<Value> lhs, shared_ptr<Value> rhs) {
    auto out = make_shared<Value>(lhs->data * rhs->data, "*", vector<shared_ptr<Value>>{lhs, rhs});
    out->_backward = [lhs, rhs, out]() {
        lhs->grad += rhs->data * out->grad;
        rhs->grad += lhs->data * out->grad;
    };
    return out;
}

inline shared_ptr<Value> operator*(shared_ptr<Value> lhs, double rhs) {
    return lhs * make_shared<Value>(rhs);
}

inline shared_ptr<Value> operator*(double lhs, shared_ptr<Value> rhs) {
    return make_shared<Value>(lhs) * rhs;
}

inline shared_ptr<Value> operator-(shared_ptr<Value> lhs, shared_ptr<Value> rhs) {
    return lhs + (rhs * -1.0);
}

inline shared_ptr<Value> operator-(shared_ptr<Value> lhs, double rhs) {
    return lhs + (-rhs);
}

inline shared_ptr<Value> operator-(double lhs, shared_ptr<Value> rhs) {
    return make_shared<Value>(lhs) - rhs;
}

inline shared_ptr<Value> pow(shared_ptr<Value> base, double exponent) {
    auto out = make_shared<Value>(std::pow(base->data, exponent), "pow", vector<shared_ptr<Value>>{base});
    out->_backward = [base, exponent, out]() {
        base->grad += exponent * std::pow(base->data, exponent - 1.0) * out->grad;
    };
    return out;
}

inline shared_ptr<Value> operator/(shared_ptr<Value> lhs, shared_ptr<Value> rhs) {
    return lhs * pow(rhs, -1.0);
}

inline shared_ptr<Value> operator/(shared_ptr<Value> lhs, double rhs) {
    return lhs * make_shared<Value>(1.0 / rhs);
}

inline shared_ptr<Value> operator/(double lhs, shared_ptr<Value> rhs) {
    return make_shared<Value>(lhs) * pow(rhs, -1.0);
}