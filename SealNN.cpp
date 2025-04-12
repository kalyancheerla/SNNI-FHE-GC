#include "seal/seal.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace seal;

template <typename T>
void print_vector(const vector<T>& vec, size_t slots_to_print = 5) {
    cout << "[";
    for (size_t i = 0; i < min(slots_to_print, vec.size()); ++i) {
        cout << vec[i];
        if (i < vec.size() - 1)
            cout << ", ";
    }
    cout << "...]" << endl;
}

void debug_print_z_output(const Ciphertext &ct,
                            Decryptor &decryptor,
                            CKKSEncoder &encoder,
                            size_t slots_to_print = 5) {
    Plaintext pt;
    decryptor.decrypt(ct, pt);

    vector<double> decoded;
    encoder.decode(pt, decoded);

    cout << "[" << endl << "0: ";
    for (size_t i = 0; i < min(slots_to_print, decoded.size()); ++i) {
        cout << decoded[i] << " ";
    }
    cout << "..." << endl << "]" << endl;
}

void debug_print_z_output(const vector<Ciphertext> &relu_vec,
                              Decryptor &decryptor,
                              CKKSEncoder &encoder,
                              size_t slots_to_print = 5) {
    cout << "[" << endl;
    for (size_t i = 0; i < relu_vec.size(); ++i) {
        Plaintext temp_plain;
        decryptor.decrypt(relu_vec[i], temp_plain);

        vector<double> decoded;
        encoder.decode(temp_plain, decoded);

        cout << i << ": ";
        for (size_t j = 0; j < min(slots_to_print, decoded.size()); ++j) {
            cout << decoded[j] << ", ";
        }
        cout << "..." << endl;
    }
    cout << "]" << endl;
}

int main() {
    // NN weights
    vector<vector<double>> w1 = {
        {0.5, -0.2, 0.1},
        {-0.3, 0.8, -0.5},
        {0.7, 0.6, -0.1},
        {-0.4, 0.2, 0.9}
    };
    vector<double> b1 = {0.1, -0.2, 0.05, 0.0};
    vector<double> w2 = {0.6, -0.4, 0.9, -0.2};
    double b2 = 0.1;

    // Collect Inputs
    vector<double> x(3);
    cout << "Enter 3 input values (floating-point): ";
    cin >> x[0] >> x[1] >> x[2];
    cout << fixed << setprecision(6);

    // Set up CKKS scheme parameters
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 30, 30}));
    double scale = pow(2.0, 30);

    // Setup Keys
    SEALContext context(parms);
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

    // Step 1 & 2
    vector<Ciphertext> z1;
    vector<Ciphertext> z2;
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
    cout << "[Debug] z1 = ";
    debug_print_z_output(z1, decryptor, encoder);
    cout << "[Debug] z2 = ";
    debug_print_z_output(z2, decryptor, encoder);

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

    cout << "[Debug] z3 = ";
    debug_print_z_output(z3, decryptor, encoder);

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

    // Decrypt Output
    Plaintext final_plain;
    decryptor.decrypt(z4, final_plain);
    vector<double> y;
    encoder.decode(final_plain, y);

    cout << endl << "[Output] z4 = " << y[0] << endl;
    return 0;
}

