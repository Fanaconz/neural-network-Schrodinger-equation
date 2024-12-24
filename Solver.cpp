#include <torch/torch.h>

#include <cmath>
#include <fstream>
#include <tuple>

using namespace torch;
using namespace std;

const int n = 20; // neurons in the hidden layer
const int m = 200; // number of examples
const double a = -3.0;
const double b = 3.0; // the interval at which we are looking for a solution
const double h = (b - a) / m; // step
Tensor ya = tensor({{1.0}});
Tensor yb = tensor({{-1.0}});
const double alpha = 0.001;
const int nepoch = 2000;

// Define a new Module.
struct Net : nn::Module {
    nn::Linear linear1{nullptr};
    nn::Linear linear2{nullptr};
    nn::Linear linear3{nullptr};
    nn::Tanh tanh;
    Tensor a;
    Tensor v;
    
    Net(const int dimension_input, const int hidden_dimension) {
        linear1 = register_module("linear1", nn::Linear(dimension_input, hidden_dimension));
        linear1->to(kDouble);
        linear2 = register_module("linear2", nn::Linear(hidden_dimension, hidden_dimension));
        linear2->to(kDouble);
        linear3 = register_module("linear3", nn::Linear(hidden_dimension, hidden_dimension)); 
        linear3->to(kDouble);
        
        tanh = register_module("tanh", nn::Tanh());

        a = register_parameter("a", torch::randn(hidden_dimension, dtype(kDouble)));
        v = register_parameter("v", torch::randn(hidden_dimension, dtype(kDouble)));
    }

    // Implement the Net's algorithm.
    Tensor forward(const Tensor &x_in) {
        Tensor x = linear1(x_in);
        x = tanh(x);
        x = linear2(x);
        x = tanh(x);
        x = linear3(x);
        
        Tensor e = torch::exp(-dot((v * x_in), (v * x_in)));
        x *= e;
        x = dot(x, a);
        return x;
    }
};

// по методу средних прямоугольников
Tensor integral(const Tensor &f, const Tensor &g) {
    Tensor sum = (f.slice(0, 0, m - 1) * g.slice(0, 0, m - 1)).sum();
    Tensor last = f[m - 1] * g[m - 1];

    return (sum - last) * h;
}

Tensor H(const Tensor &Psi, const Tensor &x) {
    auto dPsi =  autograd::grad({Psi},  {x}, {torch::ones_like(Psi)}, c10::nullopt, true)[0];
    auto d2Psi = autograd::grad({dPsi}, {x}, {torch::ones_like(Psi)}, c10::nullopt, true)[0];
    Tensor VPsi = x * x / 2 * Psi;

    return -d2Psi / 2 + VPsi;
}

Tensor loss_func(const Tensor &Psi, const Tensor &x) {
    Tensor PsiHPsi = integral(Psi, H(Psi, x));
    Tensor PsiPsi = integral(Psi, Psi);

    return PsiHPsi / PsiPsi; 
}

double E(const Tensor &Psi, const Tensor &x) {
    return loss_func(Psi, x).item<double>();
}

int main() {

    // Create a new Net.
    auto net = std::make_shared<Net>(1, n);

    optim::AdamOptions opts;
    opts.lr(alpha);
    opts.betas({0.9, 0.99});
    optim::Adam optimizer(net->parameters(), opts);

    auto x = torch::linspace(a, b, m, TensorOptions().dtype(torch::kDouble)).reshape({m, 1}); //Tensor

    cout << scientific;
    for (size_t epoch = 1; epoch <= nepoch; ++epoch) {
        x.set_requires_grad(true);
        auto y = torch::zeros({m, 1}, dtype(kDouble));
 
        for (int i = 0; i < m; i++) {
            y[i] = net->forward(x[i]);
        }
        
        Tensor loss = loss_func(y, x);

        // Reset gradients.
        optimizer.zero_grad();
        // Compute gradients of the loss w.r.t. the parameters of our model.
        loss.backward();
        // Update the parameters based on the calculated gradients.
        optimizer.step();

        if (epoch % 100 == 0) {
            cout << "Epoch = " << epoch << ", loss = " << loss.item<double>() << endl;
        }
    }

    x.set_requires_grad(true);
    auto y = torch::zeros({m, 1}, dtype(kDouble));
    
    for (int i = 0; i < m; i++) {
        y[i] = net->forward(x[i]);
    }
    
    ofstream fout("OUT.dat");
    for (int i = 0; i < m; i++)
        fout << x[i].item<double>() << "  " << y[i].item<double>() << endl;
    fout.close();

    return 0;
}

