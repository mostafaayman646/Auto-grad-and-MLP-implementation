#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <memory>
#include "Value.cpp"

using namespace std;

class Neuron {
public:
    string nonlin;
    vector<shared_ptr<Value>> W;
    shared_ptr<Value> b;
    
    Neuron(int in_dim, string nonlin = "") : nonlin(nonlin) {
        static random_device rd;  
        static mt19937 gen(rd()); 
        uniform_real_distribution<double> dis(-1.0, 1.0);
        
        for (int i = 0; i < in_dim; ++i) {
            W.push_back(make_shared<Value>(dis(gen)));
        }
        b = make_shared<Value>(dis(gen), "", vector<shared_ptr<Value>>{}, "b");
    }
    
    shared_ptr<Value> operator()(const vector<double>& X) {
        auto logit = b;
        for (size_t i = 0; i < X.size(); ++i) {
            logit = logit + (W[i] * X[i]);
        }
        if (nonlin == "Tanh") return logit->Tanh();
        if (nonlin == "Relu") return logit->Relu();
        return logit;
    }
    
    shared_ptr<Value> operator()(const vector<shared_ptr<Value>>& X) {
        auto logit = b;
        for (size_t i = 0; i < X.size(); ++i) {
            logit = logit + (W[i] * X[i]);
        }
        if (nonlin == "Tanh") return logit->Tanh();
        if (nonlin == "Relu") return logit->Relu();
        return logit;
    }
    
    vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> params = W; 
        params.push_back(b); 
        return params;
    }
};

class Layer {
public:
    vector<Neuron> neurons;
    
    Layer(int in_dim, int out_dim, string nonlin = "") {
        for (int i = 0; i < out_dim; ++i) {
            neurons.emplace_back(in_dim, nonlin);
        }
    }
    
    vector<shared_ptr<Value>> operator()(const vector<double>& X) {
        vector<shared_ptr<Value>> outs;
        for(auto& neuron : neurons) {
            outs.push_back(neuron(X)); 
        }
        return outs; 
    }
    
    vector<shared_ptr<Value>> operator()(const vector<shared_ptr<Value>>& X) {
        vector<shared_ptr<Value>> outs;
        for(auto& neuron : neurons) {
            outs.push_back(neuron(X)); 
        }
        
        return outs; 
    }
    
    vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> p_t;
        for (auto& neuron : neurons) {
            auto p = neuron.parameters();
            p_t.insert(p_t.end(), p.begin(), p.end());
        }
        return p_t;
    }
};

class MLP {
public:
    vector<Layer> layers;
    
    MLP() {}
    
    void Linear(int in_dim, int out_dim) {
        layers.emplace_back(in_dim, out_dim);
    }
    
    void Tanh() {
        if (!layers.empty()) {
            for (auto& neuron : layers.back().neurons) {
                neuron.nonlin = "Tanh";
            }
        }
    }
    
    void Relu() {
        if (!layers.empty()) {
            for (auto& neuron : layers.back().neurons) {
                neuron.nonlin = "Relu";
            }
        }
    }
    
    vector<shared_ptr<Value>> operator()(const vector<double>& X) {
        if (layers.empty()) return {};
        
        vector<shared_ptr<Value>> current_X = layers[0](X);
        for (size_t i = 1; i < layers.size(); ++i) {
            current_X = layers[i](current_X);
        }
        return current_X;
    }
    
    vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> p_t;
        for (auto& layer : layers) {
            auto p = layer.parameters();
            p_t.insert(p_t.end(), p.begin(), p.end());
        }
        return p_t;
    }
    
    void zero_grad() {
        for (auto& p : parameters()) {
            p->grad = 0.0;
        }
    }
    
    void step(double lr = 0.01) {
        for (auto& p : parameters()) {
            p->data -= lr * p->grad;
        }
    }
    
    string repr() {
        if (layers.empty()) return "MLP (Empty)";
        
        string out = "MLP (\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            Layer& layer = layers[i];
            int in_dim = layer.neurons.empty() ? 0 : layer.neurons[0].W.size();
            int out_dim = layer.neurons.size();
            string activation = layer.neurons.empty() || layer.neurons[0].nonlin == "" ? "Linear" : layer.neurons[0].nonlin;
            
            out += "  (Layer " + to_string(i) + "): [in_dim=" + to_string(in_dim) + 
                   ", out_dim=" + to_string(out_dim) + ", activation=" + activation + "]\n";
        }
        out += ") -> Total Parameters: " + to_string(parameters().size());
        return out;
    }
};