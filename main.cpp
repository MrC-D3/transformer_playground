#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

void writeMatrix(std::ostream& output, const Matrix& matrix) {
    std::size_t rows = matrix.size();
    std::size_t columns = rows == 0 ? 0 : matrix[0].size();
    output << rows << ' ' << columns << '\n';
    for (const std::vector<double>& row : matrix) {
        for (double value : row) {
            output << std::setprecision(17) << value << ' ';
        }
        output << '\n';
    }
}

bool readMatrix(std::istream& input, Matrix& matrix) {
    std::size_t rows = 0;
    std::size_t columns = 0;
    if (!(input >> rows >> columns)) {
        return false;
    }

    matrix.assign(rows, std::vector<double>(columns));
    for (std::vector<double>& row : matrix) {
        for (double& value : row) {
            if (!(input >> value)) {
                return false;
            }
        }
    }
    return true;
}

void writeVector(std::ostream& output, const std::vector<double>& values) {
    output << values.size() << '\n';
    for (double value : values) {
        output << std::setprecision(17) << value << ' ';
    }
    output << '\n';
}

bool readVector(std::istream& input, std::vector<double>& values) {
    std::size_t size = 0;
    if (!(input >> size)) {
        return false;
    }

    values.resize(size);
    for (double& value : values) {
        if (!(input >> value)) {
            return false;
        }
    }
    return true;
}

bool hasShape(const Matrix& matrix, std::size_t rows, std::size_t columns) {
    return matrix.size() == rows &&
           (rows == 0 || matrix[0].size() == columns);
}

Matrix transpose(const Matrix& matrix) {
    Matrix result(matrix[0].size(), std::vector<double>(matrix.size()));
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        for (std::size_t column = 0; column < matrix[0].size(); ++column) {
            result[column][row] = matrix[row][column];
        }
    }
    return result;
}

Matrix matmul(const Matrix& left, const Matrix& right) {
    Matrix result(left.size(), std::vector<double>(right[0].size(), 0.0));
    for (std::size_t row = 0; row < left.size(); ++row) {
        for (std::size_t shared = 0; shared < right.size(); ++shared) {
            for (std::size_t column = 0; column < right[0].size(); ++column) {
                result[row][column] += left[row][shared] * right[shared][column];
            }
        }
    }
    return result;
}

Matrix add(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    for (std::size_t row = 0; row < result.size(); ++row) {
        for (std::size_t column = 0; column < result[0].size(); ++column) {
            result[row][column] += right[row][column];
        }
    }
    return result;
}

void addInPlace(Matrix& destination, const Matrix& source) {
    for (std::size_t row = 0; row < destination.size(); ++row) {
        for (std::size_t column = 0; column < destination[0].size(); ++column) {
            destination[row][column] += source[row][column];
        }
    }
}

Matrix multiplyByScalar(const Matrix& matrix, double scalar) {
    Matrix result = matrix;
    for (std::vector<double>& row : result) {
        for (double& value : row) {
            value *= scalar;
        }
    }
    return result;
}

Matrix softmaxRows(const Matrix& scores) {
    Matrix result = scores;
    for (std::vector<double>& row : result) {
        double maximum = *std::max_element(row.begin(), row.end());
        double sum = 0.0;
        for (double& value : row) {
            value = std::exp(value - maximum);
            sum += value;
        }
        for (double& value : row) {
            value /= sum;
        }
    }
    return result;
}

Matrix layerNorm(const Matrix& input) {
    Matrix result = input;
    for (std::vector<double>& row : result) {
        double mean = 0.0;
        for (double value : row) {
            mean += value;
        }
        mean /= row.size();

        double variance = 0.0;
        for (double value : row) {
            variance += (value - mean) * (value - mean);
        }
        variance /= row.size();

        for (double& value : row) {
            value = (value - mean) / std::sqrt(variance + 1e-6);
        }
    }
    return result;
}

Matrix applyLayerNormAffine(const Matrix& normalized,
                            const std::vector<double>& gamma,
                            const std::vector<double>& beta) {
    Matrix result = normalized;
    for (std::size_t row = 0; row < result.size(); ++row) {
        for (std::size_t column = 0; column < result[0].size(); ++column) {
            result[row][column] = gamma[column] * normalized[row][column] + beta[column];
        }
    }
    return result;
}

Matrix scaleColumns(const Matrix& input, const std::vector<double>& scales) {
    Matrix result = input;
    for (std::vector<double>& row : result) {
        for (std::size_t column = 0; column < row.size(); ++column) {
            row[column] *= scales[column];
        }
    }
    return result;
}

Matrix relu(const Matrix& input) {
    Matrix result = input;
    for (std::vector<double>& row : result) {
        for (double& value : row) {
            value = std::max(0.0, value);
        }
    }
    return result;
}

Matrix makeWeights(std::size_t inputSize, std::size_t outputSize, std::mt19937& generator) {
    double limit = std::sqrt(6.0 / static_cast<double>(inputSize + outputSize));
    std::uniform_real_distribution<double> distribution(-limit, limit);
    Matrix result(inputSize, std::vector<double>(outputSize));
    for (std::size_t row = 0; row < inputSize; ++row) {
        for (std::size_t column = 0; column < outputSize; ++column) {
            result[row][column] = distribution(generator);
        }
    }
    return result;
}

Matrix positionalEncoding(Matrix input) {
    const double pi = std::acos(-1.0);
    for (std::size_t position = 0; position < input.size(); ++position) {
        for (std::size_t dimension = 0; dimension < input[0].size(); ++dimension) {
            double angle = static_cast<double>(position) /
                           std::pow(10000.0, static_cast<double>(dimension) / input[0].size());
            input[position][dimension] += dimension % 2 == 0 ? std::sin(angle) : std::cos(angle + pi / 2.0);
        }
    }
    return input;
}

Matrix layerNormBackward(const Matrix& normalized, const Matrix& gradient) {
    Matrix result = gradient;
    const double size = static_cast<double>(normalized[0].size());
    for (std::size_t row = 0; row < normalized.size(); ++row) {
        double gradientSum = 0.0;
        double normalizedGradientSum = 0.0;
        for (std::size_t column = 0; column < normalized[0].size(); ++column) {
            gradientSum += gradient[row][column];
            normalizedGradientSum += gradient[row][column] * normalized[row][column];
        }
        for (std::size_t column = 0; column < normalized[0].size(); ++column) {
            result[row][column] = (size * gradient[row][column] - gradientSum -
                                   normalized[row][column] * normalizedGradientSum) / size;
        }
    }
    return result;
}

Matrix reluBackward(const Matrix& input, const Matrix& gradient) {
    Matrix result = gradient;
    for (std::size_t row = 0; row < input.size(); ++row) {
        for (std::size_t column = 0; column < input[0].size(); ++column) {
            if (input[row][column] <= 0.0) {
                result[row][column] = 0.0;
            }
        }
    }
    return result;
}

Matrix softmaxBackwardRows(const Matrix& probabilities, const Matrix& gradient) {
    Matrix result = gradient;
    for (std::size_t row = 0; row < probabilities.size(); ++row) {
        double dotProduct = 0.0;
        for (std::size_t column = 0; column < probabilities[0].size(); ++column) {
            dotProduct += gradient[row][column] * probabilities[row][column];
        }
        for (std::size_t column = 0; column < probabilities[0].size(); ++column) {
            result[row][column] = probabilities[row][column] *
                                  (gradient[row][column] - dotProduct);
        }
    }
    return result;
}

struct EncoderCache {
    Matrix input;
    Matrix query;
    Matrix key;
    Matrix value;
    Matrix attentionScores;
    Matrix attentionProbabilities;
    Matrix context;
    Matrix attentionResidual;
    Matrix attentionNormalized;
    Matrix normalizedAttention;
    Matrix feedForwardInput;
    Matrix feedForward;
    Matrix feedForwardOutput;
    Matrix outputResidual;
    Matrix outputNormalized;
    Matrix output;
};

class Encoder {
public:
    Encoder(std::size_t modelSize, std::size_t feedForwardSize)
        : generator_(42),
          modelSize_(modelSize),
          Wq(makeWeights(modelSize, modelSize, generator_)),
          Wk(makeWeights(modelSize, modelSize, generator_)),
          Wv(makeWeights(modelSize, modelSize, generator_)),
          W1(makeWeights(modelSize, feedForwardSize, generator_)),
          W2(makeWeights(feedForwardSize, modelSize, generator_)),
          gammaAttention_(modelSize, 1.0),
          betaAttention_(modelSize, 0.0),
          gammaOutput_(modelSize, 1.0),
          betaOutput_(modelSize, 0.0) {}

    Matrix forward(const Matrix& input, EncoderCache& cache) const {
        cache.input = input;
        cache.query = matmul(input, Wq);
        cache.key = matmul(input, Wk);
        cache.value = matmul(input, Wv);

        cache.attentionScores = matmul(cache.query, transpose(cache.key));
        double scale = std::sqrt(static_cast<double>(modelSize_));
        for (std::vector<double>& row : cache.attentionScores) {
            for (double& score : row) {
                score /= scale;
            }
        }
        cache.attentionProbabilities = softmaxRows(cache.attentionScores);
        cache.context = matmul(cache.attentionProbabilities, cache.value);
        cache.attentionResidual = add(input, cache.context);
        cache.attentionNormalized = layerNorm(cache.attentionResidual);
        cache.normalizedAttention = applyLayerNormAffine(
            cache.attentionNormalized, gammaAttention_, betaAttention_);

        cache.feedForwardInput = matmul(cache.normalizedAttention, W1);
        cache.feedForward = relu(cache.feedForwardInput);
        cache.feedForwardOutput = matmul(cache.feedForward, W2);
        cache.outputResidual = add(cache.normalizedAttention, cache.feedForwardOutput);
        cache.outputNormalized = layerNorm(cache.outputResidual);
        cache.output = applyLayerNormAffine(cache.outputNormalized, gammaOutput_, betaOutput_);
        return cache.output;
    }

    void backward(const EncoderCache& cache, const Matrix& gradient, double learningRate) {
        std::vector<double> gradientGammaOutput(modelSize_, 0.0);
        std::vector<double> gradientBetaOutput(modelSize_, 0.0);
        accumulateLayerNormGradients(cache.outputNormalized, gradient,
                                      gradientGammaOutput, gradientBetaOutput);
        Matrix gradientOutputResidual = layerNormBackward(
            cache.outputNormalized, scaleColumns(gradient, gammaOutput_));
        Matrix gradientNormalizedAttention = gradientOutputResidual;
        Matrix gradientFeedForwardOutput = gradientOutputResidual;

        Matrix gradientFeedForward = matmul(gradientFeedForwardOutput, transpose(W2));
        Matrix gradientW2 = matmul(transpose(cache.feedForward), gradientFeedForwardOutput);
        Matrix gradientFeedForwardInput = reluBackward(cache.feedForwardInput, gradientFeedForward);
        Matrix gradientW1 = matmul(transpose(cache.normalizedAttention), gradientFeedForwardInput);
        addInPlace(gradientNormalizedAttention,
                   matmul(gradientFeedForwardInput, transpose(W1)));

        std::vector<double> gradientGammaAttention(modelSize_, 0.0);
        std::vector<double> gradientBetaAttention(modelSize_, 0.0);
        accumulateLayerNormGradients(cache.attentionNormalized, gradientNormalizedAttention,
                                      gradientGammaAttention, gradientBetaAttention);
        Matrix gradientAttentionResidual = layerNormBackward(
            cache.attentionNormalized,
            scaleColumns(gradientNormalizedAttention, gammaAttention_));
        Matrix gradientInput = gradientAttentionResidual;
        Matrix gradientContext = gradientAttentionResidual;

        Matrix gradientAttentionProbabilities = matmul(
            gradientContext, transpose(cache.value));
        Matrix gradientV = matmul(transpose(cache.attentionProbabilities), gradientContext);
        Matrix gradientScores = softmaxBackwardRows(
            cache.attentionProbabilities, gradientAttentionProbabilities);
        double scale = std::sqrt(static_cast<double>(modelSize_));
        gradientScores = multiplyByScalar(gradientScores, 1.0 / scale);
        Matrix gradientQ = matmul(gradientScores, cache.key);
        Matrix gradientK = matmul(transpose(gradientScores), cache.query);
        Matrix gradientWq = matmul(transpose(cache.input), gradientQ);
        Matrix gradientWk = matmul(transpose(cache.input), gradientK);
        Matrix gradientWv = matmul(transpose(cache.input), gradientV);

        addInPlace(gradientInput, matmul(gradientQ, transpose(Wq)));
        addInPlace(gradientInput, matmul(gradientK, transpose(Wk)));
        addInPlace(gradientInput, matmul(gradientV, transpose(Wv)));

        subtractScaled(W2, gradientW2, learningRate);
        subtractScaled(W1, gradientW1, learningRate);
        subtractScaled(Wq, gradientWq, learningRate);
        subtractScaled(Wk, gradientWk, learningRate);
        subtractScaled(Wv, gradientWv, learningRate);
        subtractScaled(gammaOutput_, gradientGammaOutput, learningRate);
        subtractScaled(betaOutput_, gradientBetaOutput, learningRate);
        subtractScaled(gammaAttention_, gradientGammaAttention, learningRate);
        subtractScaled(betaAttention_, gradientBetaAttention, learningRate);
    }

    bool save(std::ostream& output) const {
        output << "ENCODER\n";
        writeMatrix(output, Wq);
        writeMatrix(output, Wk);
        writeMatrix(output, Wv);
        writeMatrix(output, W1);
        writeMatrix(output, W2);
        writeVector(output, gammaAttention_);
        writeVector(output, betaAttention_);
        writeVector(output, gammaOutput_);
        writeVector(output, betaOutput_);
        return output.good();
    }

    bool load(std::istream& input) {
        std::string section;
        Matrix loadedWq;
        Matrix loadedWk;
        Matrix loadedWv;
        Matrix loadedW1;
        Matrix loadedW2;
        std::vector<double> loadedGammaAttention;
        std::vector<double> loadedBetaAttention;
        std::vector<double> loadedGammaOutput;
        std::vector<double> loadedBetaOutput;

        if (!(input >> section) || section != "ENCODER" ||
            !readMatrix(input, loadedWq) || !readMatrix(input, loadedWk) ||
            !readMatrix(input, loadedWv) || !readMatrix(input, loadedW1) ||
            !readMatrix(input, loadedW2) || !readVector(input, loadedGammaAttention) ||
            !readVector(input, loadedBetaAttention) || !readVector(input, loadedGammaOutput) ||
            !readVector(input, loadedBetaOutput)) {
            return false;
        }

        if (!hasShape(loadedWq, modelSize_, modelSize_) ||
            !hasShape(loadedWk, modelSize_, modelSize_) ||
            !hasShape(loadedWv, modelSize_, modelSize_) ||
            !hasShape(loadedW1, modelSize_, W1[0].size()) ||
            !hasShape(loadedW2, W2.size(), modelSize_) ||
            loadedGammaAttention.size() != modelSize_ ||
            loadedBetaAttention.size() != modelSize_ ||
            loadedGammaOutput.size() != modelSize_ ||
            loadedBetaOutput.size() != modelSize_) {
            return false;
        }

        Wq = loadedWq;
        Wk = loadedWk;
        Wv = loadedWv;
        W1 = loadedW1;
        W2 = loadedW2;
        gammaAttention_ = loadedGammaAttention;
        betaAttention_ = loadedBetaAttention;
        gammaOutput_ = loadedGammaOutput;
        betaOutput_ = loadedBetaOutput;
        return true;
    }

private:
    static void subtractScaled(Matrix& weights, const Matrix& gradient, double learningRate) {
        for (std::size_t row = 0; row < weights.size(); ++row) {
            for (std::size_t column = 0; column < weights[0].size(); ++column) {
                weights[row][column] -= learningRate * gradient[row][column];
            }
        }
    }

    static void accumulateLayerNormGradients(
        const Matrix& normalized,
        const Matrix& gradient,
        std::vector<double>& gradientGamma,
        std::vector<double>& gradientBeta) {
        for (std::size_t row = 0; row < normalized.size(); ++row) {
            for (std::size_t column = 0; column < normalized[0].size(); ++column) {
                gradientGamma[column] += gradient[row][column] * normalized[row][column];
                gradientBeta[column] += gradient[row][column];
            }
        }
    }

    static void subtractScaled(std::vector<double>& weights,
                               const std::vector<double>& gradient,
                               double learningRate) {
        for (std::size_t index = 0; index < weights.size(); ++index) {
            weights[index] -= learningRate * gradient[index];
        }
    }

    std::mt19937 generator_;
    std::size_t modelSize_;

public:
    Matrix Wq;
    Matrix Wk;
    Matrix Wv;
    Matrix W1;
    Matrix W2;

private:
    std::vector<double> gammaAttention_;
    std::vector<double> betaAttention_;
    std::vector<double> gammaOutput_;
    std::vector<double> betaOutput_;
};

std::vector<double> meanRows(const Matrix& matrix) {
    std::vector<double> result(matrix[0].size(), 0.0);
    for (const std::vector<double>& row : matrix) {
        for (std::size_t column = 0; column < row.size(); ++column) {
            result[column] += row[column];
        }
    }
    for (double& value : result) {
        value /= static_cast<double>(matrix.size());
    }
    return result;
}

std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> probabilities = logits;
    double maximum = *std::max_element(probabilities.begin(), probabilities.end());
    double sum = 0.0;
    for (double& value : probabilities) {
        value = std::exp(value - maximum);
        sum += value;
    }
    for (double& value : probabilities) {
        value /= sum;
    }
    return probabilities;
}

class LinearClassifier {
public:
    LinearClassifier(std::size_t inputSize, std::size_t numberOfClasses)
        : generator_(43),
          weights_(makeWeights(inputSize, numberOfClasses, generator_)),
          biases_(numberOfClasses, 0.0) {}

    double train(Encoder& encoder,
                 const Matrix& encoded,
                 const EncoderCache& encoderCache,
                 std::size_t target,
                 double learningRate) {
        std::vector<double> features = meanRows(encoded);
        std::vector<double> logits = predictLogits(features);
        std::vector<double> probabilities = softmax(logits);
        double loss = -std::log(std::max(probabilities[target], 1e-12));
        std::vector<double> gradientFeatures(features.size(), 0.0);

        for (std::size_t classIndex = 0; classIndex < biases_.size(); ++classIndex) {
            double gradient = probabilities[classIndex];
            if (classIndex == target) {
                gradient -= 1.0;
            }
            for (std::size_t feature = 0; feature < features.size(); ++feature) {
                gradientFeatures[feature] += weights_[feature][classIndex] * gradient;
                weights_[feature][classIndex] -= learningRate * features[feature] * gradient;
            }
            biases_[classIndex] -= learningRate * gradient;
        }

        Matrix gradientEncoded(encoded.size(), std::vector<double>(features.size(), 0.0));
        for (std::vector<double>& row : gradientEncoded) {
            for (std::size_t feature = 0; feature < features.size(); ++feature) {
                row[feature] = gradientFeatures[feature] /
                               static_cast<double>(encoded.size());
            }
        }
        encoder.backward(encoderCache, gradientEncoded, learningRate);
        return loss;
    }

    std::size_t predict(const Matrix& encoded) const {
        std::vector<double> logits = predictLogits(meanRows(encoded));
        return static_cast<std::size_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    bool save(std::ostream& output) const {
        output << "CLASSIFIER\n";
        writeMatrix(output, weights_);
        writeVector(output, biases_);
        return output.good();
    }

    bool load(std::istream& input) {
        std::string section;
        Matrix loadedWeights;
        std::vector<double> loadedBiases;
        if (!(input >> section) || section != "CLASSIFIER" ||
            !readMatrix(input, loadedWeights) || !readVector(input, loadedBiases)) {
            return false;
        }
        if (!hasShape(loadedWeights, weights_.size(), biases_.size()) ||
            loadedBiases.size() != biases_.size()) {
            return false;
        }

        weights_ = loadedWeights;
        biases_ = loadedBiases;
        return true;
    }

private:
    std::vector<double> predictLogits(const std::vector<double>& features) const {
        std::vector<double> logits(biases_.size(), 0.0);
        for (std::size_t feature = 0; feature < features.size(); ++feature) {
            for (std::size_t classIndex = 0; classIndex < biases_.size(); ++classIndex) {
                logits[classIndex] += features[feature] * weights_[feature][classIndex];
            }
        }
        for (std::size_t classIndex = 0; classIndex < biases_.size(); ++classIndex) {
            logits[classIndex] += biases_[classIndex];
        }
        return logits;
    }

    std::mt19937 generator_;
    Matrix weights_;
    std::vector<double> biases_;
};

void printMatrix(const Matrix& matrix) {
    for (const std::vector<double>& row : matrix) {
        for (double value : row) {
            std::cout << std::fixed << std::setprecision(3) << value << ' ';
        }
        std::cout << '\n';
    }
}

struct Sample {
    Matrix tokens;
    std::size_t label;
};

bool loadDataset(const std::string& path,
                 std::vector<Sample>& dataset,
                 std::size_t expectedFeatures) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::size_t sampleCount = 0;
    if (!(input >> sampleCount) || sampleCount == 0) {
        return false;
    }

    dataset.clear();
    dataset.reserve(sampleCount);
    for (std::size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        std::size_t label = 0;
        std::size_t tokenCount = 0;
        std::size_t featureCount = 0;
        if (!(input >> label >> tokenCount >> featureCount) ||
            tokenCount == 0 || featureCount != expectedFeatures) {
            return false;
        }

        Matrix tokens(tokenCount, std::vector<double>(featureCount));
        for (std::vector<double>& token : tokens) {
            for (double& feature : token) {
                if (!(input >> feature)) {
                    return false;
                }
            }
        }
        dataset.push_back({tokens, label});
    }
    return true;
}

void evaluate(Encoder& encoder,
              const LinearClassifier& classifier,
              const std::vector<Sample>& dataset) {
    std::size_t correct = 0;
    for (const Sample& sample : dataset) {
        EncoderCache cache;
        Matrix encoded = encoder.forward(positionalEncoding(sample.tokens), cache);
        std::size_t prediction = classifier.predict(encoded);
        correct += prediction == sample.label ? 1 : 0;
        std::cout << "Target: " << sample.label << ", predizione: " << prediction << '\n';
    }
    std::cout << "Accuratezza: " << correct << "/" << dataset.size() << '\n';
}

bool saveModel(const std::string& path,
               const Encoder& encoder,
               const LinearClassifier& classifier) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    output << "MINI_TRANSFORMER_ENCODER_V1\n";
    return encoder.save(output) && classifier.save(output);
}

bool loadModel(const std::string& path,
               Encoder& encoder,
               LinearClassifier& classifier) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::string signature;
    if (!(input >> signature) || signature != "MINI_TRANSFORMER_ENCODER_V1") {
        return false;
    }
    return encoder.load(input) && classifier.load(input);
}

void printUsage(const char* programName) {
    std::cout << "Uso:\n"
              << "  " << programName << " train    Addestra e salva il modello\n"
              << "  " << programName << " predict  Carica il modello e classifica il dataset\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2 || (std::string(argv[1]) != "train" &&
                      std::string(argv[1]) != "predict")) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    const std::string modelPath = "transformer_model.txt";
    const std::string datasetPath = mode == "train" ? "train_data.txt" : "test_data.txt";
    std::vector<Sample> dataset;
    if (!loadDataset(datasetPath, dataset, 4)) {
        std::cerr << "Errore: impossibile leggere " << datasetPath
                  << ". Controllare il formato del dataset.\n";
        return 1;
    }

    Encoder encoder(4, 8);
    LinearClassifier classifier(4, 2);

    if (mode == "train") {
        const double learningRate = 0.01;
        const std::size_t epochs = 1000;

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            for (const Sample& sample : dataset) {
                EncoderCache cache;
                Matrix encoded = encoder.forward(positionalEncoding(sample.tokens), cache);
                totalLoss += classifier.train(encoder, encoded, cache,
                                              sample.label, learningRate);
            }
            if (epoch % 200 == 0 || epoch == epochs - 1) {
                std::cout << "Epoca " << epoch + 1
                          << ", loss: " << totalLoss / dataset.size() << '\n';
            }
        }

        if (!saveModel(modelPath, encoder, classifier)) {
            std::cerr << "Errore: impossibile salvare il modello in " << modelPath << '\n';
            return 1;
        }
        std::cout << "Modello salvato in " << modelPath << '\n';
        return 0;
    }

    if (!loadModel(modelPath, encoder, classifier)) {
        std::cerr << "Errore: impossibile caricare " << modelPath
                  << ". Eseguire prima: " << argv[0] << " train\n";
        return 1;
    }
    evaluate(encoder, classifier, dataset);
}