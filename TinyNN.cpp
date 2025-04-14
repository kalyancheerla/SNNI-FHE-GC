#include "emp-tool/emp-tool.h"
#include "tinygarble/program_interface_sh.h"
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

#define FIXED_POINT_SCALE 1000.0 // 3 digit precision

void divscale_and_reduce_bitwidth(TinyGarblePI_SH* TGPI_SH,
                                  block*& output_scaled,
                                  block* input_wide,
                                  int64_t divisor,
                                  uint64_t bitwidth_input,
                                  uint64_t bitwidth_output) {
    // Create divisor block (public)
    block* divisor_block = TGPI_SH->TG_int_init(PUBLIC, bitwidth_input, divisor);

    // Allocate temp block to hold division result
    block* temp_div = TGPI_SH->TG_int(bitwidth_input);
    TGPI_SH->div(temp_div, input_wide, divisor_block, bitwidth_input);

    // Assign to reduced bitwidth
    TGPI_SH->assign(output_scaled, temp_div, bitwidth_output, bitwidth_input);

    // Cleanup
    TGPI_SH->clear_TG_int(temp_div);
}


void MyTinyNN(NetIO* io, int party) {
    TinyGarblePI_SH* TGPI_SH = new TinyGarblePI_SH(io, party);

    const int bitwidth_in = 20;   // Input and weight bitwidth
    const int bitwidth_out = 64;  // Output bitwidth

    // Input vector x
    vector<vector<int64_t>> x(1, vector<int64_t>(3));
    if (party == BOB) {
        // Get Inputs
        double x_inputs[3] = {0};
        cout << "Enter 3 input values (floating-point): ";
        cin >> x_inputs[0] >> x_inputs[1] >> x_inputs[2];

        // Scale
        for (int j = 0; j < 3; ++j)
            x[0][j] = static_cast<int64_t>(x_inputs[j] * FIXED_POINT_SCALE);

        // Print
        cout << "[Debug] Inputs received: ";
        for (int j = 0; j < 3; ++j) cout << x[0][j] << " ";
        cout << endl;
    }

    // Layer 1 weights and biases
    vector<vector<int64_t>> w1(3, vector<int64_t>(4));
    vector<int64_t> b1(4);
    vector<vector<int64_t>> w2(4, vector<int64_t>(1));
    int64_t b2 = 0;

    if (party == ALICE) {
        double w1_vals[4][3] = {{0.5, -0.2, 0.1}, {-0.3, 0.8, -0.5}, {0.7, 0.6, -0.1}, {-0.4, 0.2, 0.9}};
        double b1_vals[4] = {0.1, -0.2, 0.05, 0.0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                w1[i][j] = static_cast<int64_t>(w1_vals[j][i] * FIXED_POINT_SCALE);
        for (int i = 0; i < 4; ++i)
            b1[i] = static_cast<int64_t>(b1_vals[i] * FIXED_POINT_SCALE);

        // w1 & b1
        cout << "[Debug] Layer 1: Weights and Biases: " << endl;
        for (int i = 0; i < 4; ++i) {
            cout << i << ": ";
            for (int j = 0; j < 3; ++j)
                cout << w1[j][i] << " ";
            cout << " | Bias: " << b1[i] << endl;
        }

        double w2_vals[4] = {0.6, -0.4, 0.9, -0.2};
        double b2_val = 0.1;
        for (int i = 0; i < 4; ++i)
            w2[i][0] = static_cast<int64_t>(w2_vals[i] * FIXED_POINT_SCALE);
        b2 = static_cast<int64_t>(b2_val * FIXED_POINT_SCALE);

        cout << "[Debug] Layer 2: Weights and Bias: ";
        for (int i = 0; i < 4; ++i)
            cout << w2[i][0] << " ";
        cout << "| Bias: " << b2 << endl;
    }

//    cout << "[1 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;
    // Encrypt inputs and weights
    auto x_enc  = TGPI_SH->TG_int_init(BOB, bitwidth_in, x, 1, 3);
    auto w1_enc = TGPI_SH->TG_int_init(ALICE, bitwidth_in, w1, 3, 4);
    auto b1_enc = TGPI_SH->TG_int_init(ALICE, bitwidth_in, b1, 4);
    auto w2_enc = TGPI_SH->TG_int_init(ALICE, bitwidth_in, w2, 4, 1);
    auto b2_enc = TGPI_SH->TG_int_init(ALICE, bitwidth_in, b2);

    cout << "[2 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;
    TGPI_SH->gen_input_labels();
    cout << "[3 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;
    TGPI_SH->retrieve_input_vector_labels(x_enc,  BOB, bitwidth_in, 1, 3);
    TGPI_SH->retrieve_input_vector_labels(w1_enc, ALICE, bitwidth_in, 3, 4);
    TGPI_SH->retrieve_input_vector_labels(b1_enc, ALICE, bitwidth_in, 4);
    TGPI_SH->retrieve_input_vector_labels(w2_enc, ALICE, bitwidth_in, 4, 1);
    TGPI_SH->retrieve_input_labels(b2_enc, ALICE, bitwidth_in);
    TGPI_SH->clear_input_labels();
//    cout << "[4 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // Matrix multiplication using mat_mult (outputs are 64-bit wide)
    auto z1_enc = TGPI_SH->TG_int(bitwidth_out, 1, 4);
    TGPI_SH->mat_mult(1, 3, 4, x_enc, w1_enc, z1_enc, 0, bitwidth_in, bitwidth_in, bitwidth_out, bitwidth_out);
//    cout << "[5 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    auto z1_bias = TGPI_SH->TG_int(bitwidth_in, 1, 4);
    for (int j = 0; j < 4; ++j) {
        // Down scale after multiplication
        //TGPI_SH->right_shift(z1_enc[0][j], 10, bitwidth_out); //2^10 = 1024 so, approx. /1000
        divscale_and_reduce_bitwidth(TGPI_SH, z1_bias[0][j], z1_enc[0][j], FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);

        // Add bias & perform ReLU
        TGPI_SH->add(z1_bias[0][j], z1_bias[0][j], b1_enc[j], bitwidth_in, bitwidth_in);
        TGPI_SH->relu(z1_bias[0][j], bitwidth_in);
    }
//    cout << "[6 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // Reveal outputs
    cout << "[Debug] z2 = [" << endl;
    for (int j = 0; j < 4; ++j) {
        int64_t out_j = TGPI_SH->reveal(z1_bias[0][j], bitwidth_in);
        cout << j << ": " << out_j << " " << (double(out_j) / FIXED_POINT_SCALE) << endl;
    }
    cout << "]" << endl;
//    cout << "[7 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // Again Mat Mult
    auto z2_enc = TGPI_SH->TG_int(bitwidth_out, 1, 1);
    TGPI_SH->mat_mult(1, 4, 1, z1_bias, w2_enc, z2_enc, 0, bitwidth_in, bitwidth_in, bitwidth_out, bitwidth_out);
//    cout << "[8 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // Down scale after multiplication
    auto z2_enc_scaled = TGPI_SH->TG_int(bitwidth_in, 1, 1);
    divscale_and_reduce_bitwidth(TGPI_SH, z2_enc_scaled[0][0], z2_enc[0][0], FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);
//    cout << "[9 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

//    int64_t out_temp = TGPI_SH->reveal(z2_enc_scaled[0][0], bitwidth_in);
//    cout << "Neuron 0: " << out_temp << " " << (double(out_temp) / FIXED_POINT_SCALE) << endl;

    TGPI_SH->add(z2_enc_scaled[0][0], z2_enc_scaled[0][0], b2_enc, bitwidth_in, bitwidth_in);

//    cout << "[10 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;
    int64_t out_j = TGPI_SH->reveal(z2_enc_scaled[0][0], bitwidth_in);
    cout << "[Debug] z3 = [" << out_j << " " << (double(out_j) / FIXED_POINT_SCALE) << "]" << endl;
//    cout << "[11 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // sigmoid using poly approx.
    int64_t c0 = static_cast<int64_t>(0.5  * FIXED_POINT_SCALE);
    int64_t c1 = static_cast<int64_t>(0.197 * FIXED_POINT_SCALE);
    int64_t c2 = static_cast<int64_t>(0.004 * FIXED_POINT_SCALE);

    // Compute z²
    auto z2_sq = TGPI_SH->TG_int(bitwidth_out);
    TGPI_SH->mult(z2_sq, z2_enc_scaled[0][0], z2_enc_scaled[0][0], bitwidth_in);
//    cout << "[12 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // Down scale after multiplication
    auto z2_sq_scaled = TGPI_SH->TG_int(bitwidth_in);
    divscale_and_reduce_bitwidth(TGPI_SH, z2_sq_scaled, z2_sq, FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);
//    cout << "[13 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // term1 = c1 * z
    auto term1 = TGPI_SH->TG_int(bitwidth_out);
    TGPI_SH->mult(term1, z2_enc_scaled[0][0], c1, bitwidth_in, bitwidth_in);
//    cout << "[14 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    auto term1_scaled = TGPI_SH->TG_int(bitwidth_in);
    divscale_and_reduce_bitwidth(TGPI_SH, term1_scaled, term1, FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);
//    cout << "[15 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // term2 = c2 * z²
    auto term2 = TGPI_SH->TG_int(bitwidth_out);
    TGPI_SH->mult(term2, z2_sq_scaled, c2, bitwidth_in, bitwidth_in);
//    cout << "[16 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    auto term2_scaled = TGPI_SH->TG_int(bitwidth_in);
    divscale_and_reduce_bitwidth(TGPI_SH, term2_scaled, term2, FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);
//    cout << "[17 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // term1 - term2
    auto sigmoid_out = TGPI_SH->TG_int(bitwidth_in);
    TGPI_SH->sub(sigmoid_out, term1_scaled, term2_scaled, bitwidth_in, bitwidth_in);
//    cout << "[18 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // Add constant c0
    TGPI_SH->add(sigmoid_out, sigmoid_out, c0, bitwidth_in);

//    cout << "[19 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;
    // Reveal result
    int64_t out_val = TGPI_SH->reveal(sigmoid_out, bitwidth_in);
    cout << "[Output] z4 = " << out_val << " " << (double(out_val) / FIXED_POINT_SCALE) << endl;
//    cout << "[20 " << (party == ALICE ? "ALICE" : "BOB") << "] Rounds: " << io->comm_rounds << "(" << io->previous_op << ")" << endl;

    // Cleanup
    TGPI_SH->clear_TG_int(x_enc, 1, 3);
    TGPI_SH->clear_TG_int(w1_enc, 3, 4);
    TGPI_SH->clear_TG_int(b1_enc, 4);
    TGPI_SH->clear_TG_int(w2_enc, 4, 1);
    TGPI_SH->clear_TG_int(b2_enc);
    TGPI_SH->clear_TG_int(z1_enc, 1, 4);
    TGPI_SH->clear_TG_int(z1_bias, 1, 4);
    TGPI_SH->clear_TG_int(z2_enc, 1, 1);
    TGPI_SH->clear_TG_int(z2_enc_scaled, 1, 1);

    TGPI_SH->clear_TG_int(z2_sq);
    TGPI_SH->clear_TG_int(z2_sq_scaled);
    TGPI_SH->clear_TG_int(term1);
    TGPI_SH->clear_TG_int(term1_scaled);
    TGPI_SH->clear_TG_int(term2_scaled);
    TGPI_SH->clear_TG_int(sigmoid_out);

    delete TGPI_SH;
}

int main(int argc, char** argv) {
    int party = 1, port = 1234;
    string server_ip;

    po::options_description desc{"Allowed options"};
    desc.add_options()
        ("help,h", "produce help message")
        ("party,k", po::value<int>(&party)->default_value(1), "party id: 1 for garbler, 2 for evaluator")
        ("port,p", po::value<int>(&port)->default_value(1234), "socket port")
        ("server_ip,s", po::value<string>(&server_ip)->default_value("127.0.0.1"), "server's IP address");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            cout << desc << endl;
            return 0;
        }
        po::notify(vm);
    } catch (po::error& e) {
        cout << "ERROR: " << e.what() << "\n" << desc << endl;
        return -1;
    }

    NetIO* io = new NetIO(party == ALICE ? nullptr : server_ip.c_str(), port, true);
    io->set_nodelay();

    MyTinyNN(io, party);

    delete io;
    return 0;
}
