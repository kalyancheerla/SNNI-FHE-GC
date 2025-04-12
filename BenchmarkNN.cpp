#include <seal/seal.h>
#include <emp-tool/emp-tool.h>
#include <tinygarble/program_interface_sh.h>
#include <boost/program_options.hpp>
#include <sstream>
#include <chrono>
#include <vector>
#include <iostream>
#include <sys/resource.h>

using namespace std;
using namespace seal;
using namespace emp;
namespace po = boost::program_options;


size_t getMaxRSS() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024L; // Convert from KB to bytes
}

void Test(NetIO* io, int party) {
    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();

    if (party == ALICE) {
        string msg = "Hello"; // 5 bytes
        uint32_t len = msg.size();
        io->send_data(&len, sizeof(len));
        io->send_data(msg.data(), len);

        uint32_t reply_len;
        io->recv_data(&reply_len, sizeof(reply_len));
        string reply(reply_len, 0);
        io->recv_data(&reply[0], reply_len);
        cout << "[ALICE] Received reply: " << reply << endl;

    } else {
        uint32_t len;
        io->recv_data(&len, sizeof(len));
        string msg(len, 0);
        io->recv_data(&msg[0], len);
        cout << "[BOB] Received message: " << msg << endl;

        string reply = "Test"; // 4 bytes
        uint32_t reply_len = reply.size();
        io->send_data(&reply_len, sizeof(reply_len));
        io->send_data(reply.data(), reply_len);
    }

    auto end = Clock::now();
    chrono::duration<double> diff = end - start;
    cout << "[TEST] Round-trip time: " << diff.count() << " seconds " << endl;
    cout << "[MEM] Peak memory usage: " << getMaxRSS() / 1024.0 / 1024.0 << " MB" << endl;
    cout << "[" << (party == ALICE ? "ALICE" : "BOB") << "] Sent: " << (float)(io->counter) << " bytes" << endl;
}



void PlainEvaluation(NetIO* io, int party) {
    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();

    if (party == ALICE) {
        // Alice receives input
        uint32_t len;
        io->recv_data(&len, sizeof(len));
        vector<double> x(len);
        io->recv_data(x.data(), len * sizeof(double));

        vector<vector<double>> w1 = {
            {0.5, -0.2, 0.1},
            {-0.3, 0.8, -0.5},
            {0.7, 0.6, -0.1},
            {-0.4, 0.2, 0.9}
        };
        vector<double> b1 = {0.1, -0.2, 0.05, 0.0};
        vector<double> w2 = {0.6, -0.4, 0.9, -0.2};
        double b2 = 0.1;

        vector<double> z2(4);
        for (int i = 0; i < 4; i++) {
            double sum = b1[i];
            for (int j = 0; j < 3; j++)
                sum += w1[i][j] * x[j];
            z2[i] = max(0.0, sum);
        }

        double z3 = b2;
        for (int i = 0; i < 4; i++)
            z3 += w2[i] * z2[i];
        double y = 1.0 / (1.0 + exp(-z3));

        io->send_data(&y, sizeof(double));

    } else {
       // Bob sends input vector
        vector<double> input = {1.0, 2.0, 3.0};
        uint32_t len = input.size();
        io->send_data(&len, sizeof(len));
        io->send_data(input.data(), len * sizeof(double));

        // Bob receives
        double result;
        io->recv_data(&result, sizeof(double));
        cout << "[BOB] Output y: " << result << endl;
    }

    auto end = Clock::now();
    chrono::duration<double> diff = end - start;
    cout << "[PLAIN] Round-trip time: " << diff.count() << " seconds " << endl;
    cout << "[MEM] Peak memory usage: " << getMaxRSS() / 1024.0 / 1024.0 << " MB" << endl;
    cout << "[" << (party == ALICE ? "ALICE" : "BOB") << "] Sent: " << (float)(io->counter) << " bytes" << endl;
}



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


void GCEvaluation(NetIO* io, int party) {
    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();

    TinyGarblePI_SH* TGPI_SH = new TinyGarblePI_SH(io, party);

    const int bitwidth_in = 20;   // Input and weight bitwidth
    const int bitwidth_out = 64;  // Output bitwidth

    // Input vector x
    vector<vector<int64_t>> x(1, vector<int64_t>(3));
    if (party == BOB) {
        // Get Inputs
        vector<double> x_inputs = {1.0, 2.0, 3.0};

        // Scale
        for (int j = 0; j < 3; ++j)
            x[0][j] = static_cast<int64_t>(x_inputs[j] * FIXED_POINT_SCALE);
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

        double w2_vals[4] = {0.6, -0.4, 0.9, -0.2};
        double b2_val = 0.1;
        for (int i = 0; i < 4; ++i)
            w2[i][0] = static_cast<int64_t>(w2_vals[i] * FIXED_POINT_SCALE);
        b2 = static_cast<int64_t>(b2_val * FIXED_POINT_SCALE);
    }

    // Encrypt inputs and weights
    auto x_enc  = TGPI_SH->TG_int_init(BOB, bitwidth_in, x, 1, 3);
    auto w1_enc = TGPI_SH->TG_int_init(ALICE, bitwidth_in, w1, 3, 4);
    auto b1_enc = TGPI_SH->TG_int_init(ALICE, bitwidth_in, b1, 4);
    auto w2_enc = TGPI_SH->TG_int_init(ALICE, bitwidth_in, w2, 4, 1);
    auto b2_enc = TGPI_SH->TG_int_init(ALICE, bitwidth_in, b2);

    TGPI_SH->gen_input_labels();
    TGPI_SH->retrieve_input_vector_labels(x_enc,  BOB, bitwidth_in, 1, 3);
    TGPI_SH->retrieve_input_vector_labels(w1_enc, ALICE, bitwidth_in, 3, 4);
    TGPI_SH->retrieve_input_vector_labels(b1_enc, ALICE, bitwidth_in, 4);
    TGPI_SH->retrieve_input_vector_labels(w2_enc, ALICE, bitwidth_in, 4, 1);
    TGPI_SH->retrieve_input_labels(b2_enc, ALICE, bitwidth_in);
    TGPI_SH->clear_input_labels();

    // Matrix multiplication using mat_mult (outputs are 64-bit wide)
    auto z1_enc = TGPI_SH->TG_int(bitwidth_out, 1, 4);
    TGPI_SH->mat_mult(1, 3, 4, x_enc, w1_enc, z1_enc, 0, bitwidth_in, bitwidth_in, bitwidth_out, bitwidth_out);

    auto z1_bias = TGPI_SH->TG_int(bitwidth_in, 1, 4);
    for (int j = 0; j < 4; ++j) {
        // Down scale after multiplication
        divscale_and_reduce_bitwidth(TGPI_SH, z1_bias[0][j], z1_enc[0][j], FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);

        // Add bias & perform ReLU
        TGPI_SH->add(z1_bias[0][j], z1_bias[0][j], b1_enc[j], bitwidth_in, bitwidth_in);
        TGPI_SH->relu(z1_bias[0][j], bitwidth_in);
    }

    // Again Mat Mult
    auto z2_enc = TGPI_SH->TG_int(bitwidth_out, 1, 1);
    TGPI_SH->mat_mult(1, 4, 1, z1_bias, w2_enc, z2_enc, 0, bitwidth_in, bitwidth_in, bitwidth_out, bitwidth_out);

    // Down scale after multiplication
    auto z2_enc_scaled = TGPI_SH->TG_int(bitwidth_in, 1, 1);
    divscale_and_reduce_bitwidth(TGPI_SH, z2_enc_scaled[0][0], z2_enc[0][0], FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);

    TGPI_SH->add(z2_enc_scaled[0][0], z2_enc_scaled[0][0], b2_enc, bitwidth_in, bitwidth_in);

    // sigmoid using poly approx.
    int64_t c0 = static_cast<int64_t>(0.5  * FIXED_POINT_SCALE);
    int64_t c1 = static_cast<int64_t>(0.197 * FIXED_POINT_SCALE);
    int64_t c2 = static_cast<int64_t>(0.004 * FIXED_POINT_SCALE);

    // Compute z²
    auto z2_sq = TGPI_SH->TG_int(bitwidth_out);
    TGPI_SH->mult(z2_sq, z2_enc_scaled[0][0], z2_enc_scaled[0][0], bitwidth_in);

    // Down scale after multiplication
    auto z2_sq_scaled = TGPI_SH->TG_int(bitwidth_in);
    divscale_and_reduce_bitwidth(TGPI_SH, z2_sq_scaled, z2_sq, FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);

    // term1 = c1 * z
    auto term1 = TGPI_SH->TG_int(bitwidth_out);
    TGPI_SH->mult(term1, z2_enc_scaled[0][0], c1, bitwidth_in, bitwidth_in);

    auto term1_scaled = TGPI_SH->TG_int(bitwidth_in);
    divscale_and_reduce_bitwidth(TGPI_SH, term1_scaled, term1, FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);

    // term2 = c2 * z²
    auto term2 = TGPI_SH->TG_int(bitwidth_out);
    TGPI_SH->mult(term2, z2_sq_scaled, c2, bitwidth_in, bitwidth_in);

    auto term2_scaled = TGPI_SH->TG_int(bitwidth_in);
    divscale_and_reduce_bitwidth(TGPI_SH, term2_scaled, term2, FIXED_POINT_SCALE, bitwidth_out, bitwidth_in);

    // term1 - term2
    auto sigmoid_out = TGPI_SH->TG_int(bitwidth_in);
    TGPI_SH->sub(sigmoid_out, term1_scaled, term2_scaled, bitwidth_in, bitwidth_in);

    // Add constant c0
    TGPI_SH->add(sigmoid_out, sigmoid_out, c0, bitwidth_in);

    // Reveal result
    int64_t out_val = TGPI_SH->reveal(sigmoid_out, bitwidth_in);
    cout << "[BOTH] Output y: " << out_val << " " << (double(out_val) / FIXED_POINT_SCALE) << endl;

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

    auto end = Clock::now();
    chrono::duration<double> diff = end - start;
    cout << "[GC] Round-trip time: " << diff.count() << " seconds " << endl;
    cout << "[MEM] Peak memory usage: " << getMaxRSS() / 1024.0 / 1024.0 << " MB" << endl;
    cout << "[" << (party == ALICE ? "ALICE" : "BOB") << "] Sent: " << (float)(io->counter)/1024 << " KB" << " (" << io->counter << "B)" << endl;
}



// Helper: Send a SEAL object as string
template<typename T>
void send_seal_obj(NetIO* io, T& obj) {
    stringstream ss;
    obj.save(ss);
    string s = ss.str();
    uint32_t len = s.size();
    io->send_data(&len, sizeof(len));
    io->send_data(s.data(), len);
}

// Helper: Receive a SEAL object from string
template<typename T>
void recv_seal_obj(NetIO* io, T& obj, SEALContext context) {
    uint32_t len;
    io->recv_data(&len, sizeof(len));
    string s(len, 0);
    io->recv_data(&s[0], len);
    stringstream ss(s);
    obj.load(context, ss);
}

// Helper: Send and receive ciphertext
void send_ciphertext(NetIO* io, Ciphertext& ct) {
    stringstream ss;
    ct.save(ss);
    string s = ss.str();
    uint32_t len = s.size();
    io->send_data(&len, sizeof(len));
    io->send_data(s.data(), len);
}

void recv_ciphertext(NetIO* io, Ciphertext& ct, SEALContext context) {
    uint32_t len;
    io->recv_data(&len, sizeof(len));
    string s(len, 0);
    io->recv_data(&s[0], len);
    stringstream ss(s);
    ct.load(context, ss);
}

void FHEPolynomialEvaluation(NetIO* io, int party) {
    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();

    size_t poly_modulus_degree = 16384;
    double scale = pow(2.0, 30);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 30, 30}));

    SEALContext context(parms);

    if (party == BOB) {
        // Collect Inputs
        vector<double> x = {1.0, 2.0, 3.0};

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        keygen.create_public_key(public_key);
        RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);
        GaloisKeys galois_keys;
        keygen.create_galois_keys(galois_keys);

        // Setup Functions
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, secret_key);
        Evaluator evaluator(context);
        CKKSEncoder encoder(context);

        // Slot inputs
        size_t slot_count = encoder.slot_count(); // #slots = 8192 = 16384/2
        vector<double> x_slots(slot_count, 0.0);
        for (int i = 0; i < 3; i++)
            x_slots[i] = x[i];

        Plaintext plain_x;
        encoder.encode(x_slots, scale, plain_x);
        Ciphertext enc_x;
        encryptor.encrypt(plain_x, enc_x);

        // Send public key, relin, galois, encrypted input
        send_seal_obj(io, public_key);
        send_seal_obj(io, relin_keys);
        send_seal_obj(io, galois_keys);
        send_ciphertext(io, enc_x);

        // Receive result and decrypt
        Ciphertext ct_result;
        recv_ciphertext(io, ct_result, context);

        Plaintext pt_result;
        decryptor.decrypt(ct_result, pt_result);
        vector<double> result;
        encoder.decode(pt_result, result);

        cout << "[BOB] Output y: " << result[0] << endl;
    }
    else {
        vector<vector<double>> w1 = {
            {0.5, -0.2, 0.1},
            {-0.3, 0.8, -0.5},
            {0.7, 0.6, -0.1},
            {-0.4, 0.2, 0.9}
        };
        vector<double> b1 = {0.1, -0.2, 0.05, 0.0};
        vector<double> w2 = {0.6, -0.4, 0.9, -0.2};
        double b2 = 0.1;

        // Alice side: receive keys and input
        PublicKey pk;
        RelinKeys relin_keys;
        GaloisKeys galois_keys;
        recv_seal_obj(io, pk, context);
        recv_seal_obj(io, relin_keys, context);
        recv_seal_obj(io, galois_keys, context);

        Encryptor encryptor(context, pk);
        Evaluator evaluator(context);
        CKKSEncoder encoder(context);

        Ciphertext enc_x;
        recv_ciphertext(io, enc_x, context);

        // Step 1 & 2
        vector<Ciphertext> z1;
        vector<Ciphertext> z2;
        size_t slot_count = encoder.slot_count(); // #slots = 8192 = 16384/2
        for (int i = 0; i < 4; ++i) {
            vector<double> weight_row(slot_count, 0.0);
            for (int j = 0; j < 3; j++)
                weight_row[j] = w1[i][j];
            //print_vector(weight_row);

            Plaintext plain_wi;
            encoder.encode(weight_row, scale, plain_wi);
            evaluator.mod_switch_to_inplace(plain_wi, enc_x.parms_id());

            Ciphertext temp;
            evaluator.multiply_plain(enc_x, plain_wi, temp);
            //evaluator.rescale_to_next_inplace(temp);
            //cout << "[Debug] temp (after rescale) scale = " << log2(temp.scale()) << ", level = " << context.get_context_data(temp.parms_id())->chain_index() << endl;

            Ciphertext rotated1, rotated2;
            evaluator.rotate_vector(temp, 1, galois_keys, rotated1);
            evaluator.rotate_vector(temp, 2, galois_keys, rotated2);
            evaluator.add_inplace(temp, rotated1);
            evaluator.add_inplace(temp, rotated2);

            Plaintext plain_bi;
            encoder.encode(b1[i], temp.scale(), plain_bi);
            evaluator.mod_switch_to_inplace(plain_bi, temp.parms_id());
            evaluator.add_plain_inplace(temp, plain_bi);
            evaluator.rescale_to_next_inplace(temp);
            z1.push_back(temp);

            Ciphertext relu_z;
            evaluator.square(temp, relu_z);
            evaluator.relinearize_inplace(relu_z, relin_keys);
            evaluator.rescale_to_next_inplace(relu_z);
            z2.push_back(relu_z);
        }

        // Step 3
        Ciphertext z3;
        for (int i = 0; i < 4; ++i) {
            Plaintext wi_plain;
            encoder.encode(w2[i], z2[i].scale(), wi_plain);
            evaluator.mod_switch_to_inplace(wi_plain, z2[i].parms_id());

            Ciphertext term;
            evaluator.multiply_plain(z2[i], wi_plain, term);

            if (i == 0) {
                z3 = term;  // initialize with first term
            } else {
                evaluator.add_inplace(z3, term);
            }
        }

        Plaintext b2_plain;
        encoder.encode(b2, z3.scale(), b2_plain);
        evaluator.mod_switch_to_inplace(b2_plain, z3.parms_id());
        evaluator.add_plain_inplace(z3, b2_plain);

        // Step 4
        Ciphertext z3_sq;
        evaluator.square(z3, z3_sq);
        evaluator.relinearize_inplace(z3_sq, relin_keys);
        evaluator.rescale_to_next_inplace(z3_sq);

        evaluator.mod_switch_to_inplace(z3, z3_sq.parms_id());

        Plaintext const_0_197, const_neg_0_004;
        encoder.encode(0.197, z3.scale(), const_0_197);
        encoder.encode(-0.004, z3_sq.scale(), const_neg_0_004);
        evaluator.mod_switch_to_inplace(const_0_197, z3.parms_id());
        evaluator.mod_switch_to_inplace(const_neg_0_004, z3.parms_id());

        // Multiply terms
        Ciphertext term1, term2;
        evaluator.multiply_plain(z3, const_0_197, term1);
        evaluator.rescale_to_next_inplace(term1);

        evaluator.multiply_plain(z3_sq, const_neg_0_004, term2);
        evaluator.rescale_to_next_inplace(term2);

        // Align all scales and levels
        evaluator.mod_switch_to_inplace(term1, term2.parms_id());
        Plaintext const_0_5;
        encoder.encode(0.5, term1.scale(), const_0_5);
        evaluator.mod_switch_to_inplace(const_0_5, term2.parms_id());

        // Add all terms
        Ciphertext z4;
        evaluator.add_plain(term1, const_0_5, z4);
        term2.scale() = z4.scale();
        evaluator.add_inplace(z4, term2);

        // Send back result
        send_ciphertext(io, z4);
    }

    auto end = Clock::now();
    chrono::duration<double> diff = end - start;
    cout << "[FHE] Round-trip time: " << diff.count() << " seconds " << endl;
    cout << "[MEM] Peak memory usage: " << getMaxRSS() / 1024.0 / 1024.0 << " MB" << endl;
    cout << "[" << (party == ALICE ? "ALICE" : "BOB") << "] Sent: " << (float)(io->counter)/1024 << " KB" << " (" << io->counter << "B)" << endl;
}



int main(int argc, char** argv) {
    int party = 1, port = 1234;
    string server_ip = "127.0.0.1", program = "FHE";

    po::options_description desc{"Allowed options"};
    desc.add_options()
        ("help,h", "produce help message")
        ("party,k", po::value<int>(&party)->default_value(1), "party id: 1 for Alice, 2 for Bob")
        ("port,p", po::value<int>(&port)->default_value(1234), "socket port")
        ("server_ip,s", po::value<string>(&server_ip)->default_value("127.0.0.1"), "server's IP address")
        ("program,f", po::value<string>(&program)->default_value("PLAIN"), "Function to execute: PLAIN or GC or FHE or TEST");

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

    if (program == "PLAIN")
        PlainEvaluation(io, party);
    else if (program == "GC")
        GCEvaluation(io, party);
    else if (program == "FHE")
        FHEPolynomialEvaluation(io, party);
    else if (program == "TEST")
        Test(io, party);
    else
        cerr << "Unknown program option: " << program << endl;

    delete io;
    return 0;
}
