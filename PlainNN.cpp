#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// Exact sigmoid: 1/(1+e^-x)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Polynomial approx. of sigmoid: 0.5 + 0.197x - 0.004x^2
double approx_sigmoid(double x) {
    return 0.5 + 0.197 * x - 0.004 * x * x;
}

// Exact ReLU: max(0,x)
double relu(double x) {
    return max(0.0, x);
}

// Polynomial approx. of ReLU: x^2
double approx_relu(double x) {
    return x * x;
}

// toggle 0: Exact ReLU & sigmoid functions
// toggle 1: Exact ReLU & approx. sigmoid functions
// toggle 2: Approx. ReLU & sigmoid functions
double evaluate_nn(vector<vector<double>> w1, vector<double> b1,
                   vector<double> w2, double b2,
                   vector<double> x, int toggle) {
    vector<double> z1(4);
    vector<double> z2(4);

    for (int i = 0; i < 4; i++) {
        double sum = b1[i];
        for (int j = 0; j < 3; j++)
            sum += w1[i][j] * x[j];
        z1[i] = sum;
        if (toggle == 0 || toggle == 1)
            z2[i] = relu(sum);
        else
            z2[i] = approx_relu(sum);
    }

    cout << "Step 1 - z1 = w1·x+b1 = ";
    for (double val : z1) cout << val << " ";
    cout << endl;

    cout << "Step 2 - z2 = ReLU(z0) = ";
    for (double val : z2) cout << val << " ";
    cout << endl;

    double z3 = b2;
    cout << "Step 3 - z3 = w2·z2+b2 = ";
    for (int i = 0; i < 4; i++) {
        z3 += w2[i] * z2[i];
    }
    cout << z3 << endl;

    double z4 = 0;
    if (toggle == 0)
        z4 = sigmoid(z3);
    else
        z4 = approx_sigmoid(z3);
    cout << "Step 4 - z4 = sigmoid(z3) = " << z4 << endl;
    return z4;
}


int main() {
    vector<vector<double>> w1 = {
        {0.5, -0.2, 0.1},
        {-0.3, 0.8, -0.5},
        {0.7, 0.6, -0.1},
        {-0.4, 0.2, 0.9}
    };
    vector<double> b1 = {0.1, -0.2, 0.05, 0.0};
    vector<double> w2 = {0.6, -0.4, 0.9, -0.2};
    double b2 = 0.1;

    vector<double> x(3);
    cout << "Enter 3 input values (floating-point): ";
    cin >> x[0] >> x[1] >> x[2];
    cout << fixed << setprecision(6);

    cout << endl;
    cout << "################################################################################" << endl;
    cout << "[NN evaluation with exact ReLU & sigmoid - Start]" << endl;
    evaluate_nn(w1, b1, w2, b2, x, 0);
    cout << "[End]" << endl;
    cout << "################################################################################" << endl;
    cout << endl << endl;

    cout << "################################################################################" << endl;
    cout << "[NN evaluation with exact ReLU & approx. sigmoid - Start]" << endl;
    evaluate_nn(w1, b1, w2, b2, x, 1);
    cout << "[End]" << endl;
    cout << "################################################################################" << endl;
    cout << endl << endl;

    cout << "################################################################################" << endl;
    cout << "[NN evaluation with approx. ReLU & sigmoid - Start]" << endl;
    evaluate_nn(w1, b1, w2, b2, x, 2);
    cout << "[End]" << endl;
    cout << "################################################################################" << endl;
    cout << endl;

    return 0;
}
