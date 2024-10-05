#include<iostream>
#include<vector>
#include<math.h>
#include<random>
#include <chrono>
using namespace std;

constexpr auto TANH = 0;
constexpr auto SIGMOID = 1;
constexpr auto RELU = 2;
constexpr auto LRELU = 3;
constexpr auto SWISH = 4;
constexpr auto SOFTPLUS = 5;
constexpr auto SOFTSIGN = 6;
constexpr auto BENT_IDENTITY = 7;
constexpr auto HARDTANH = 8;
constexpr auto SELU = 9;
constexpr auto ELU = 10;
constexpr auto SOFTMAX = 11;


constexpr auto MSE = 0;
constexpr auto BCEL = 1;
constexpr auto CBCEL = 2;
constexpr auto CCE = 3;
constexpr auto HUBER = 4;
constexpr auto FOCAL = 5;
constexpr auto DICE = 6;


static int networkLayerSize;
static int inputLayersize;
static int hiddenLayersize;
static int outputLayerSize;






class Neuron {
public:
	Neuron() {};

	void makeConnections(int numberOfNextLayerNeurons);


	// temp gets
	vector<double>& getweightsAndConnections() { return w; }
	double getInput() { return x_ActivatedInput; }
	double getBais() { return b_Bais; }
	double getWeightedSum() { return weightedSum; }
	vector<double>& getVelocities()
	{
		return velocities;
	}

	//another thing
	// temp sets
	void set_xInput(double x) { x_ActivatedInput = x; }
	void resetX() { x_ActivatedInput = 0; }
	void setWeightedSum(double w) { weightedSum = w; }


private:
	//construction of activation(w*x(wx+wx+wx)+b);
	vector<double> w; //weights
	double x_ActivatedInput; // X = activated(w+w2+w3+w4...wn)x0
	double b_Bais; // bias
	double weightedSum;//same as y! or output
	vector<double> velocities;

	

};
void Neuron::makeConnections(int numberOfNextLayerNeurons)
{
	random_device rd;
	mt19937 random_engine(rd());
	uniform_real_distribution<> dist(-1.0, 1.0);

	w = vector<double>(numberOfNextLayerNeurons);

	for (auto& tempCon : w)
	{
		tempCon = dist(random_engine);
	}
	

	b_Bais = dist(random_engine);
	//b_Bais = 0; //change
}
class Network {

public:
	Network(int numberOfInputLayerNeuron, int numberOfHiddenLayers, int numberOfNeurons, int numberOfOutputNeurons);
	void feedForward();
	double getWeightedSum(int layerIndex, int NueronIndex);
	void feedForwardFlexableOutputs(int WhichActivationFunc = SIGMOID);
	void feedForwardFlexableInputs();



	//activations

	double ActivationFunctionsChoice(int Choice, double weightedSum);
	double derivativeOfActivationFunction(int Choice, double weightedSum);

	//tanh
	double tanhActivationFunction(double weightedSum) { return tanh(weightedSum); }
	double derivateOfTanhActivationFunction(double weightedSum){ return 1 - (tanh(weightedSum) * tanh(weightedSum));}
	
	//sigmoid
	double sigmoidActivationFucntion(double weightedSum) {return 1.0 / (1.0 + exp(-weightedSum));}
	double derivativeOfSigmoidFucntion( double weightedSum) {	double sigmoid = sigmoidActivationFucntion(weightedSum);	return sigmoid * (1.0 - sigmoid);}

	//relu
	double reluActivationFunction(double x) { return (x > 0) ? x : 0; }
	double reluActivationFunction_derivative(double x) { return (x > 0) ? 1 : 0;  }

	// Leaky ReLU
	double leaky_reluActivationFunction(double x, double alpha) {	return (x > 0) ? x : alpha * x;}
	double leaky_reluActivationFunction_derivative(double x, double alpha) {	return (x > 0) ? 1 : alpha;}

	// Swish
	double swishActivationFunction(double x) { return x / (1 + exp(-x));}
	double swishActivationFunction_derivative(double x) {double exp_x = exp(-x); double denominator = (1 + exp_x) * (1 + exp_x); return (exp_x * (x + 1) - 1) / denominator;}

	// Softmax
	double softmax(double x) {	double exp_x = exp(x); return exp_x / (1 + exp_x); }
	double softmax_derivative(double x) {		double exp_x = exp(x);	double denominator = (1 + exp_x) * (1 + exp_x);		return exp_x / denominator;	}

	// Softplus
	double softplus(double x) {		return log(1 + exp(x));	}
	double softplus_derivative(double x) {		return 1 / (1 + exp(-x));	}

	// Softsign
	double softsign(double x) {	return x / (1 + abs(x));}
	double softsign_derivative(double x) {		double denominator = (1 + abs(x)) * (1 + abs(x));		return 1 / denominator;	}

	// Bent Identity
	double bent_identity(double x) {		return (sqrt(x * x + 1) - 1) / 2 + x;	}
	double bent_identity_derivative(double x) {	return x / sqrt(x * x + 1) + 1;}

	// Hardtanh
	double hardtanh(double x) {		if (x < -1) return -1;		if (x > 1) return 1;		return x;	}
	double hardtanh_derivative(double x) {		if (x < -1 || x > 1) return 0;		return 1;	}

	// SELU
	double selu(double x) {		if (x < 0) return 1.0507 * exp(x) - 1.0507;	return 1.0507 * x;}
	double selu_derivative(double x) {		if (x < 0) return 1.0507 * exp(x);		return 1.0507;	}

	// ELU
	double elu(double x) {		if (x < 0) return exp(x) - 1;		return x;	}
	double elu_derivative(double x) {		if (x < 0) return exp(x);		return 1;	}





	//costs
	void costFunctionChoice(int costFunctionChoice);

	void derivativeOfCostFunctionChoice(int WhichCostFunction);

	void MSEcostFunction();
	void derivativeOfMSEcostFunction();

	void BinaryCrossEntropyLoss();
	void DerivativeOfBinaryCrossEntropyLoss();

	void ClipedBinaryCrossEntropyLoss();
	void DerivativeOfClipedBinaryCrossEntropyLoss();
	//new
	void CategoricalCrossEntropyLoss();
	void DerivativeOfCategoricalCrossEntropyLoss();

	void HuberLoss(double delta);
	void DerivativeOfHuberLoss(double delta);

	void FocalLoss(double alpha, double gamma);
	void DerivativeOfFocalLoss(double alpha, double gamma);

	void DiceLoss();
	void DerivativeOfDiceLoss();

	void backPropagationCalculateGradient();
	void backPropagationPropagate();
	


	//temp gets
	vector<vector<Neuron>>& getLayers(void) { return Layers; }
	vector<double> getOutput() { return Output; }
	vector<double> getInput() { return Input; }
	double getCost() { return cost; }
	vector<double> getDeltacosts() { return deltaCosts; }
	vector<vector<vector<double>>>& getCalculatedGradient() { return calculatedGradient; }

	// temp sets
	void setInput(vector<double> input) { Input = input; }
	void setOutput(vector<double> output) { Output = output; }
	void allocGradientAndChained();
	void resetOutput() { Output.clear(); }
	void resetInput() { Input.clear(); }
	void resetDesired() { desiredOutput.clear(); }
	void setDesiredOutput(vector<double> desire) { desiredOutput = desire; }
	void resetCostsDeltas() { deltaCosts.clear(); cost = 0.0; }
	
	void resetGradient() { calculatedGradient.clear(); }
	void resetChained() { chainedStored.clear(); }





	//public variables
	int networkLayerSize;
	int inputLayersize;
	int hiddenLayersize;
	int outputLayerSize;

	double sign(double x) {
		if (x > 0) {
			return 1.0;
		}
		else if (x < 0) {
			return -1.0;
		}
		else {
			return 0.0;
		}
	}


private:



	vector<vector<vector<double>>> calculatedGradient;
	vector < vector <double>> chainedStored;
	vector<vector<Neuron>> Layers;
	vector<double> Input;
	vector<double> Output;
	vector<double> desiredOutput;

	double etaLearningRate = 0.2;

	double cost;
	vector<double> deltaCosts;



	//static variables ADJUSTABLE!! here
	// 
		//for huber
	double delta = 0.5;//tolerance adjust as needed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
	//for focal
	double alpha = 0.5, gamma = 0.5; // adjust as needed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	//leaky
	double alphaLeaky = 0.1;


};
void Network::allocGradientAndChained()
{
	
	calculatedGradient = vector<vector<vector<double>>>(networkLayerSize-1);//gradient allocation
	chainedStored = vector<vector<double>> (networkLayerSize); // chainedstore allocatio
	// Allocate first layer
	calculatedGradient[0].resize(inputLayersize, vector<double>(hiddenLayersize));
	chainedStored[0].resize(inputLayersize);
	// Allocate hidden layers
	for (int i = 1; i < networkLayerSize - 1; i++) {
		if (i == networkLayerSize - 2) {
			calculatedGradient[i].resize(hiddenLayersize, vector<double>(outputLayerSize));
			chainedStored[i].resize(hiddenLayersize);
		}
		else {
			calculatedGradient[i].resize(hiddenLayersize, vector<double>(hiddenLayersize));
			chainedStored[i].resize(hiddenLayersize);
		}
	}
	// Allocate last layer
	//calculatedGradient[networkLayerSize - 1].resize(outputLayerSize, vector<double>(1)); changed
	chainedStored[networkLayerSize - 1].resize(outputLayerSize);
}
Network::Network(int numberOfInputLayersNeuron, int numberOfHiddenLayers, int numberOfHiddenNeurons, int numberOfOutputNuerons)
{
	/*
	* initialization
	*/

	networkLayerSize = 1 + numberOfHiddenLayers + 1;
	inputLayersize = numberOfInputLayersNeuron;
	hiddenLayersize = numberOfHiddenNeurons;
	outputLayerSize = numberOfOutputNuerons;

	cost = 0.0;
	etaLearningRate = 0.0;
	
	//allocation-for-Gradient-and-Chain-Vectors
	allocGradientAndChained();

	vector<vector<Neuron>> tempLayer(networkLayerSize);//supparated the inputlayer and the outputlayers

	//inputLayer made
	for (int countInputLayers = 0; countInputLayers < numberOfInputLayersNeuron; countInputLayers++)
	{
		Neuron tempNeuron = Neuron();
		tempNeuron.makeConnections(numberOfHiddenNeurons);
		tempLayer[0].push_back(tempNeuron);
	}

	//hiddenLayer made
	for (int countLayers = 1; countLayers <= numberOfHiddenLayers; countLayers++)
	{
		for (int countNeurons = 0; countNeurons < numberOfHiddenNeurons; countNeurons++)
		{
			Neuron tempNeuron = Neuron();
			tempNeuron.makeConnections(numberOfHiddenNeurons);
			tempLayer[countLayers].push_back(tempNeuron);
		}
	}

	//outputLayer made
	for (int outputNeurons = 0; outputNeurons < numberOfOutputNuerons; outputNeurons++) {
		Neuron tempNeuron = Neuron();
		//tempNeuron.makeConnections(1);
		tempLayer[numberOfHiddenLayers + 1].push_back(tempNeuron);//+2 because we have added extra 2 conteners for the input and output!, the input took the [0], the out put took[+1]
	}
	Layers = tempLayer;
}
double Network::getWeightedSum(int layerIndex, int neuronIndex)
{
	/*
	* takes LayerIndex = Layer weights to be extracted from
	* takes neuronIndex = neuron weights to be extracted for
	* for every neuron in layer sums the weights for neuron
	* returns the weighted sum
	*/

	double tempWeightedSum = 0.0;
	for (auto& nueron : Layers[layerIndex])
	{
		double weight = nueron.getweightsAndConnections()[neuronIndex];//weight
		double x = nueron.getInput();//activated
		double b = nueron.getBais();//bias
		tempWeightedSum = tempWeightedSum + (weight * x + b);
	}

	return tempWeightedSum;
}
double Network::ActivationFunctionsChoice(int Choice, double weightedSum)
{
	
	/*
	* takes Choice function
	* takes WeightedSum
	* Activates weightedsum By Choice
	* returns Activated
	*/
	double Activated = 0.0;
	switch (Choice){

	case TANH:
		Activated = tanhActivationFunction(weightedSum);
		break;

	case SIGMOID:
		Activated = sigmoidActivationFucntion(weightedSum);
		break;

	case RELU:
		Activated = reluActivationFunction(weightedSum);
		break;

	case LRELU:
		Activated = leaky_reluActivationFunction(weightedSum, alphaLeaky);
		break;

	case SWISH:
		Activated = swishActivationFunction(weightedSum);
		break;

	case SOFTPLUS:
		Activated = softplus(weightedSum);
		break;

	case SOFTSIGN:
		Activated = softsign(weightedSum);
		break;

	case BENT_IDENTITY:
		Activated = bent_identity(weightedSum);
		break;

	case HARDTANH:
		Activated = hardtanh(weightedSum);
		break;

	case SELU:
		Activated = selu(weightedSum);
		break;

	case ELU:
		Activated = elu(weightedSum);
		break;

	case SOFTMAX:
		Activated = softmax(weightedSum);
		break;

	default:
		cerr << "***Activation Function Jumped!!***\n\a";
		break;
	}

	return Activated;
}
double Network::derivativeOfActivationFunction(int Choice, double weightedSum)
{

	/*
	* takes Choice function
	* takes WeightedSum
	* Activates weightedsum By Choice
	* returns Derivatives
	*/
	double Activated = 0.0;
	switch (Choice) {

	case TANH:
		Activated = derivateOfTanhActivationFunction(weightedSum);
		break;

	case SIGMOID:
		Activated = derivativeOfSigmoidFucntion(weightedSum);
		break;

	case RELU:
		Activated = reluActivationFunction_derivative(weightedSum);
		break;

	case LRELU:
		Activated = leaky_reluActivationFunction_derivative(weightedSum, alphaLeaky);
		break;

	case SWISH:
		Activated = swishActivationFunction_derivative(weightedSum);
		break;

	case SOFTPLUS:
		Activated = softplus_derivative(weightedSum);
		break;

	case SOFTSIGN:
		Activated = softsign_derivative(weightedSum);
		break;

	case BENT_IDENTITY:
		Activated = bent_identity_derivative(weightedSum);
		break;

	case HARDTANH:
		Activated = hardtanh_derivative(weightedSum);
		break;

	case SELU:
		Activated = selu_derivative(weightedSum);
		break;

	case ELU:
		Activated = elu_derivative(weightedSum);
		break;

	case SOFTMAX:
		Activated = softmax_derivative(weightedSum);
		break;

	default:
		cerr << "***Activation Derivative Function Jumped!!***\n\a";
		break;
	}

	return Activated;
}

void Network::feedForwardFlexableOutputs(int functionChoice)
{
	/*
	* it takes FunctionChoice
	* goes through lastLayers
	* gets weightedsum
	* sends it to be activateed
	* as many times as there are outputs
	*/
	double weightedSum;
	int singleLayer = Layers.size() - 2; //-2 because its coming from hiddenl layer
	for (int singleNeuron = 0; singleNeuron < Layers[singleLayer + 1].size(); singleNeuron++)//for every output
	{
		weightedSum = getWeightedSum(singleLayer, singleNeuron); //sends the layer and the index number of the next nueron for the weight to be extracted for
		Layers[singleLayer + 1][singleNeuron].setWeightedSum(weightedSum);
		double Activated = ActivationFunctionsChoice(functionChoice, weightedSum);		
		Output.push_back(Activated);
	}
}
void Network::feedForwardFlexableInputs()
{
	/*
	* sets input for every neuron in input Layer
	* sets weighted sum, which is just the input, or none
	*/
	int size = Layers.front().size();
	auto& inputNeuron = Layers.front();

	for (int firstLayerNeurons = 0; firstLayerNeurons < size; firstLayerNeurons++)
	{
		inputNeuron[firstLayerNeurons].set_xInput(Input[firstLayerNeurons]);
		//calculate weighted sum, the weighted sum is the input its self.
		inputNeuron[firstLayerNeurons].setWeightedSum(Input[firstLayerNeurons]);//could be left none
	}

}
void Network::feedForward()
{

	/*
	* Feed Forward
	* calls set input.
	*/
	//intput refract here so that its flexable
	feedForwardFlexableInputs();

	double weightedSum;
	for (int singleLayer = 0; singleLayer < Layers.size() - 2; singleLayer++)//changed to -2
	{			
			for (int singleNeuron = 0; singleNeuron < Layers[singleLayer + 1].size(); singleNeuron++)
			{
				weightedSum = getWeightedSum(singleLayer, singleNeuron);
				Layers[singleLayer + 1][singleNeuron].setWeightedSum(weightedSum);

				double Activated = ActivationFunctionsChoice(TANH, weightedSum);

				Layers[singleLayer + 1][singleNeuron].set_xInput(Activated);

			}
		
	}

	//output with choiceActivationFunction
	feedForwardFlexableOutputs(SIGMOID);//change

}
void Network::costFunctionChoice(int costFunctionChoice)
{


	switch (costFunctionChoice)
	{
	case MSE:
		MSEcostFunction();
		break;
	case BCEL:
		BinaryCrossEntropyLoss();
		break;
	case CBCEL:
		ClipedBinaryCrossEntropyLoss();
		break;
	case CCE:
		CategoricalCrossEntropyLoss();
		break;
	case HUBER:
		HuberLoss(delta);
		break;
	case FOCAL:
		FocalLoss(alpha, gamma);
		break;
	case DICE:
		DiceLoss();
		break;

	default:
		cerr << "\n***Cost Function Jumped!!****\n\a";
		break;
	}
}
void Network::derivativeOfCostFunctionChoice(int WhichCostFunction)
{

	switch (WhichCostFunction)
	{
	case MSE:
		derivativeOfMSEcostFunction();
		break;
	case BCEL:
		DerivativeOfBinaryCrossEntropyLoss();
		break;
	case CBCEL:
		DerivativeOfClipedBinaryCrossEntropyLoss();
		break;
	case CCE:
		DerivativeOfCategoricalCrossEntropyLoss();
		break;
	case HUBER:
		DerivativeOfHuberLoss(delta);
		break;
	case FOCAL:
		DerivativeOfFocalLoss(alpha, gamma);
		break;
	case DICE:
		DerivativeOfDiceLoss();
		break;

	default:
		cerr << "\n***Cost Derivative Function Jumped!!****\n\a";
		break;
	}
}
void Network::MSEcostFunction()
{
	int sameIndex = 0;
	cost = 0;
	for (auto& singleOutputNeurons : Output)
	{
		cost += pow(singleOutputNeurons - desiredOutput[sameIndex], 2);
		sameIndex++;
	}
	cost /= Output.size();

}
void Network::derivativeOfMSEcostFunction()
{
	int sameIndex = 0;
	deltaCosts.resize(Output.size());
	for (auto& singleOutputNeurons : Output)
	{
		deltaCosts[sameIndex] = 2 * (singleOutputNeurons - desiredOutput[sameIndex]);
		sameIndex++;
	}
}

//costs
void Network::BinaryCrossEntropyLoss()
{
	int sameIndex = 0;
	cost = 0;


	for (auto& singleOutputNeurons : Output)
	{

		cost -= (desiredOutput[sameIndex] * log(singleOutputNeurons) + ((1 - desiredOutput[sameIndex]) * log(1 - singleOutputNeurons)));

		sameIndex++;
	}
	cost /= Output.size();

}
void Network::DerivativeOfBinaryCrossEntropyLoss()
{
	int sameIndex = 0;
	deltaCosts.clear();
	for (auto& singleOutputNeurons : Output)
	{
		deltaCosts.push_back(-(desiredOutput[sameIndex] / singleOutputNeurons + 1e-10) - ((1 - desiredOutput[sameIndex]) / ((1 - singleOutputNeurons) + 1e-10)));
		sameIndex++;
	}
}
void Network::ClipedBinaryCrossEntropyLoss()
{
	int sameIndex = 0;
	cost = 0;
	for (auto& singleOutputNeurons : Output)
	{
		double clipedOutSafeValue = max(min(singleOutputNeurons, 1 - 1e-10), 1e-10);
		cost -= (desiredOutput[sameIndex] * log(clipedOutSafeValue) + (1 - desiredOutput[sameIndex] * log(1 - clipedOutSafeValue)));
		sameIndex++;
	}
	cost /= Output.size();
}
void Network::DerivativeOfClipedBinaryCrossEntropyLoss()
{
	int sameIndex = 0;
	for (auto& singleOutputNeurons : Output)
	{
		// Clipping the output to avoid log(0) or division by zero
		double clipedOutSafeValue = max(min(singleOutputNeurons, 1 - 1e-10), 1e-10);

		// Calculate the gradient of the cost w.r.t each output neuron
		deltaCosts[sameIndex] = -(desiredOutput[sameIndex] / clipedOutSafeValue) +
			((1 - desiredOutput[sameIndex]) / (1 - clipedOutSafeValue));

		sameIndex++;
	}
}
void Network::CategoricalCrossEntropyLoss() {
	int sameIndex = 0;
	cost = 0;


	for (auto& singleOutputNeurons : Output)
	{
		cost -= desiredOutput[sameIndex] * log(singleOutputNeurons + 1e-10);
		sameIndex++;
	}
	cost /= Output.size();
}
void Network::DerivativeOfCategoricalCrossEntropyLoss() {
	int sameIndex = 0;
	deltaCosts.clear();
	for (auto& singleOutputNeurons : Output)
	{
		deltaCosts.push_back(-desiredOutput[sameIndex] / (singleOutputNeurons + 1e-10));
		sameIndex++;
	}
}
void Network::HuberLoss(double delta) {

	int sameIndex = 0;

	cost = 0;
	for (auto& singleOutputNeurons : Output) {
		double error = singleOutputNeurons - desiredOutput[sameIndex];
		if (abs(error) <= delta) {
			cost += 0.5 * error * error;
		}
		else {
			cost += delta * (abs(error) - 0.5 * delta);
		}
		sameIndex++;
	}
	cost /= Output.size();
}
void Network::DerivativeOfHuberLoss(double delta) {
	int sameIndex = 0;
	deltaCosts.clear();
	for (auto& singleOutputNeurons : Output) {
		double error = singleOutputNeurons - desiredOutput[sameIndex];
		if (abs(error) <= delta) {
			deltaCosts.push_back(error);
		}
		else {
			deltaCosts.push_back(delta * sign(error));
		}
		sameIndex++;
	}
}
void Network::FocalLoss(double alpha, double gamma) {
	int sameIndex = 0;
	cost = 0;

	for (auto& singleOutputNeurons : Output) {
		double pt = singleOutputNeurons;
		cost -= alpha * pow(1 - pt, gamma) * desiredOutput[sameIndex] * log(pt + 1e-10) +
			(1 - alpha) * pow(pt, gamma) * (1 - desiredOutput[sameIndex]) * log(1 - pt + 1e-10);
		sameIndex++;
	}
	cost /= Output.size();
}
void Network::DerivativeOfFocalLoss(double alpha, double gamma) {
	int sameIndex = 0;
	deltaCosts.clear();
	for (auto& singleOutputNeurons : Output) {
		double pt = singleOutputNeurons;
		deltaCosts.push_back(alpha * pow(1 - pt, gamma) * desiredOutput[sameIndex] / (pt + 1e-10) -
			(1 - alpha) * pow(pt, gamma) * (1 - desiredOutput[sameIndex]) / (1 - pt + 1e-10));
		sameIndex++;
	}
}
void Network::DiceLoss() {
	int sameIndex = 0;
	double intersection = 0;
	double sum = 0;

	for (auto& singleOutputNeurons : Output) {
		intersection += singleOutputNeurons * desiredOutput[sameIndex];
		sum += pow(singleOutputNeurons, 2) + pow(desiredOutput[sameIndex], 2);
		sameIndex++;
	}
	cost = 1 - (2 * intersection) / (sum + 1e-10);
}
void Network::DerivativeOfDiceLoss() {
	int sameIndex = 0;
	deltaCosts.clear();
	double intersection = 0;
	double sum = 0;

	for (auto& singleOutputNeurons : Output) {
		intersection += singleOutputNeurons * desiredOutput[sameIndex];
		sum += pow(singleOutputNeurons, 2) + pow(desiredOutput[sameIndex], 2);
		sameIndex++;
	}

	for (auto& singleOutputNeurons : Output) {
		deltaCosts.push_back((2 * desiredOutput[sameIndex] * (sum - 2 * intersection)) / (sum * sum + 1e-10));
	}
}
void Network::backPropagationCalculateGradient()
{
	/*
	* calculates the chain of last layer which is cost/output same as (delta) * output/derivativeOfActivationFunction(weightedsum from backLayer).
	* there is no gradient for the last
	* the chain is calculated inside the first for
	*
	* goes to the second large loop and for every front neuron,(assuning the derivativeActivation(weightedsum) is chained already
	* it calculates the chained*weightedsum of back Neuron; iterates throught the front neurons, while storing the chain in tempchain, the gradient is stored.
	* at the end it stores the chained for that back neuron index
	*/

	int indexOfLayer = Layers.size()-1;
	int indexOfFrontNeuron;
	int indexOfBackNeuron;


	// gradient for last layer;
	int indexOfLastNeuron = 0;
	for (auto deltacost : deltaCosts)
	{
		double x = Layers[indexOfLayer][indexOfLastNeuron].getWeightedSum();
		chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivativeOfActivationFunction(SIGMOID, x));//output activation sigmoid
		indexOfLastNeuron++;
		//double w = Layers[indexOfLayer][indexOfLastNeuron].getweightsAndConnections()[0];
		//double b = Layers[indexOfLayer][indexOfLastNeuron].getBais();
		//double weightedSumOfTheOutput = w * x + b;
		//chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivativeOfSigmoidFucntion(w * x + b));     //temporary chain
		//these 3 were for the last incorrect layer changed.
		//calculatedGradient[indexOfLayer][indexOfLastNeuron][0] = chainedStored[indexOfLayer][indexOfLastNeuron] * x;
		//chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivativeOfSigmoidFucntion(w * x + b) * w * derivateOfTanhActivationFunction(x)); // chained stored
		//chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivateOfTanhActivationFunction(w * x + b) * w * derivateOfTanhActivationFunction(x)); // remove
	}



	for (int currentLayerIndex = indexOfLayer; currentLayerIndex > 0; currentLayerIndex--)
	{
		indexOfBackNeuron = 0;//leave me alone
		//back neuron
		for (int backSize = Layers[currentLayerIndex - 1].size(); indexOfBackNeuron < backSize; indexOfBackNeuron++)
		{
			indexOfFrontNeuron = 0;//leave me alone

			Neuron& backLayerNeuron = Layers[currentLayerIndex - 1][indexOfBackNeuron];
			vector<double>& backWieght = backLayerNeuron.getweightsAndConnections(); //w
			double backWeightedSum = backLayerNeuron.getWeightedSum(); //x

			double tempChained = 0.0;
			//front nueron
			for (int frontSize = Layers[currentLayerIndex].size(); indexOfFrontNeuron < frontSize; indexOfFrontNeuron++)
			{
				Neuron& FrontLayerNeuron = Layers[currentLayerIndex][indexOfFrontNeuron];
				calculatedGradient[currentLayerIndex - 1][indexOfBackNeuron][indexOfFrontNeuron] = chainedStored[currentLayerIndex][indexOfFrontNeuron]/*derivative of thanh(x)*/ * backWeightedSum/* x of current*/; //optimize backweight is there
				tempChained += (chainedStored[currentLayerIndex][indexOfFrontNeuron] * backWieght[indexOfFrontNeuron] * derivativeOfActivationFunction(TANH, backWeightedSum));
			}
			chainedStored[currentLayerIndex - 1][indexOfBackNeuron] = tempChained;
		}
	}
}


//done




























void Network::backPropagationPropagate()
{
	/*
	* the gradient is subtracted from the old weight
	*/


	int netsize = calculatedGradient.size();
	//iterate layers
	for (int numberofLayer = 0; numberofLayer < netsize; numberofLayer++)
	{
		//cout << "Layer::::::" << numberofLayer << endl;
		int neuroSize = calculatedGradient[numberofLayer].size();
		//iterate neurons
		for (int numberOfneuron = 0; numberOfneuron < neuroSize; numberOfneuron++)
		{
			//cout << "Neuron:::::::" << numberOfneuron << endl;
			int weightsize = calculatedGradient[numberofLayer][numberOfneuron].size();
			auto& Layerweights = Layers[numberofLayer][numberOfneuron].getweightsAndConnections();
			//iterate weights
			//cout << "***start weight***\n";
			for (int numberOfWeight = 0; numberOfWeight < weightsize; numberOfWeight++)
			{
				//cout << " \n = old*****************: " << Layerweights[numberOfWeight] << endl;
				//cout << "[" << numberofLayer << "][" << numberOfneuron << "][" << numberOfWeight << "]" << endl;// << Layerweights[numberOfWeight] << " + " << calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight];
				//cout << "******** the gradient: " << calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight] << endl;
				
				//(calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight] < 0 && Layers[numberofLayer][numberOfneuron].getweightsAndConnections()[numberOfWeight] < 0)?  Layers[numberofLayer][numberOfneuron].getweightsAndConnections()[numberOfWeight] += -1*calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight] : Layers[numberofLayer][numberOfneuron].getweightsAndConnections()[numberOfWeight] += calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight];//change
				//cout << " \n = updated*****************: " << Layerweights[numberOfWeight]<<endl;
				if (desiredOutput[0] == 1)
				{
					Layerweights[numberOfWeight] -= calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight]/100;
				}
				else {
					Layerweights[numberOfWeight] += calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight]/100;//change +
				}
				

			}
			//cout << "***endl weight***\n";
		}
	}
}

pair<vector<vector<double>>, vector<double>> trainingGen(int size = 1000)
{
	pair<vector<vector<double>>, vector<double>> pairedData;
	vector<vector<double>> tempDataInput;
	vector<double> tempDataDesire;

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> distrib(0, 1); // Use binary distribution for XOR-like data

	for (int a = 0; a < size; ++a) {
		// Generate random inputs
		int x1 = distrib(gen);
		int x2 = distrib(gen);
		// XOR logic for generating output
		double desired_output = (x1 != x2) ? 1 : 0;

		tempDataInput.push_back({ static_cast<double>(0), static_cast<double>(0) });
		tempDataDesire.push_back(0);
		tempDataInput.push_back({ static_cast<double>(0), static_cast<double>(1) });
		tempDataDesire.push_back(1);
		tempDataInput.push_back({ static_cast<double>(1), static_cast<double>(1) });
		tempDataDesire.push_back(0);
		tempDataInput.push_back({ static_cast<double>(1), static_cast<double>(1) });
		tempDataDesire.push_back(0);
		tempDataInput.push_back({ static_cast<double>(0), static_cast<double>(0) });
		tempDataDesire.push_back(0);
		tempDataInput.push_back({ static_cast<double>(1), static_cast<double>(0) });
		tempDataDesire.push_back(1);


	}


	pairedData.first = tempDataInput;
	pairedData.second = tempDataDesire;

	return pairedData;


	//remove
	if (size == 100 || size == 1000 || size == 10)
	{
		for (int a = 0; a < size / 2; ++a) {
			// Generate random inputs
			int x1 = distrib(gen);
			int x2 = distrib(gen);
			// XOR logic for generating output
			double desired_output = (x1 != x2) ? 1 : 0;

			// Push the generated pair into vectors
			//		tempDataInput.push_back({ static_cast<double>(x1), static_cast<double>(x2) });
			//      tempDataDesire.push_back(desired_output);

			tempDataInput.push_back({ static_cast<double>(0.001), static_cast<double>(0.001) });//remove
			tempDataDesire.push_back(0.001);
			tempDataInput.push_back({ static_cast<double>(0.999), static_cast<double>(0.999) });//remove
			tempDataDesire.push_back(0.001);

			//tempDataInput.push_back({ static_cast<double>(1), static_cast<double>(1) });//remove
			//tempDataDesire.push_back(-1);


		}
	}
	else {

		for (int a = 0; a < size / 2; ++a) {
			// Generate random inputs
			int x1 = distrib(gen);
			int x2 = distrib(gen);
			// XOR logic for generating output
			double desired_output = (x1 != x2) ? 1 : 0;

			// Push the generated pair into vectors
			//		tempDataInput.push_back({ static_cast<double>(x1), static_cast<double>(x2) });
			//      tempDataDesire.push_back(desired_output);

			tempDataInput.push_back({ static_cast<double>(0.001), static_cast<double>(0.999) });//remove
			tempDataDesire.push_back(0.999);

			tempDataInput.push_back({ static_cast<double>(0.999), static_cast<double>(0.001) });//remove
			tempDataDesire.push_back(0.999);



		}
	}

}

int main()
{
	auto start = chrono::high_resolution_clock::now();//-------------------------------------------------------------------time cost calculation
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	//cout << "Time taken: " << duration.count() << " microseconds" << endl;





	//network creation




	Network tempNet = Network(2, 3, 4, 1); //************************************************add a way to choose hidden layer activation and output layer activation, cost too!!!!!!!













	
	auto& Layers = tempNet.getLayers();

	cout << "*******************************WEIGHTS****************************\n";
	//get weights
	int i = 0, j = 0, k = 0;
	for (auto& neurons : Layers)
	{
		cout << "\n******LAYER********: " << i << endl;
		for (auto& n : neurons)
		{
			cout << "****NEURON****: " << j << endl;
			auto& currentConnections = n.getweightsAndConnections();
			for (auto& w : currentConnections)
			{
				cout << "**WEIGHT**: [" << i << "] [" << j << "] " << "[" << k << "] = " << w << endl;
				k++;
			}
			k = 0;
			j++;
		}
		j = 0;
		i++;
	}





	//gettraining data;

	int epoch = 50000;
	auto pairedInputAndDesire = trainingGen(epoch);




	cout << "\n*********TESS***********\n";
	int ask = 0;
	while (ask != 99)
	{
		int input;
		vector<double> tempinputVec;
		cout << "\nstart: ";
		cin >> input;
		tempinputVec.push_back(input);
		cin >> input;
		tempinputVec.push_back(input);

		tempNet.setInput(tempinputVec);

		tempNet.feedForward();

		cout << tempNet.getOutput()[0];

		tempNet.resetOutput();
		tempNet.resetDesired();
		tempNet.resetInput();

		cout << "\ncontinue?: ";
		cin >> ask;
	}






	//test for right changes
	vector<vector<double>> input = pairedInputAndDesire.first;
	vector<double> desire = pairedInputAndDesire.second;

	double tempinput = 0.0;
	int count = 0;
	while (count < epoch)
	{

		//separate learning
		tempNet.setInput(input[count]);
		tempNet.setDesiredOutput(vector<double>{desire[count]});

		//step #2 feedforward.
		tempNet.feedForward();

		//step #3 calculate cost  
		tempNet.costFunctionChoice(BCEL);

		//step #4 calculate cost/output
		tempNet.derivativeOfCostFunctionChoice(BCEL);

		//steo #5 do a gradientcalculation
		tempNet.backPropagationCalculateGradient();

		//step #6 do a back prop
		tempNet.backPropagationPropagate();



		cout << "input 1st : " << input[count].front();
		//cin >> tempinput;
		cout << endl;
		cout << "input 2nd : " << input[count].back();
		//cin >> tempinput;
		cout << endl;
		cout << "desire : " << desire[count];
		//cin >> tempinput;
		cout << endl;
		cout << "output: " << tempNet.getOutput()[0] << " & cost: " << tempNet.getCost() << endl;









		tempNet.resetOutput();
		tempNet.resetDesired();
		tempNet.resetInput();
		tempNet.resetCostsDeltas();
		tempNet.resetChained();
		tempNet.resetGradient();
		tempNet.allocGradientAndChained();
		count++;

	}

	cout << "\n*********TESS***********\n";
	ask = 0;
	while (ask != 99)
	{
		int input;
		vector<double> tempinputVec;
		cout << "\nstart: ";
		cin >> input;
		tempinputVec.push_back(input);
		cin >> input;
		tempinputVec.push_back(input);

		tempNet.setInput(tempinputVec);

		tempNet.feedForward();

		cout << tempNet.getOutput()[0];

		tempNet.resetOutput();
		tempNet.resetDesired();
		tempNet.resetInput();

		cout << "\ncontinue?: ";
		cin >> ask;
	}

	

	cout << "*******************************WEIGHTS****************************\n";
	//get weights
	i = 0, j = 0, k = 0;
	for (auto& neurons : Layers)
	{
		cout << "\n******LAYER********: " << i << endl;
		for (auto& n : neurons)
		{
			cout << "****NEURON****: " << j << endl;
			auto& currentConnections = n.getweightsAndConnections();
			for (auto& w : currentConnections)
			{
				cout << "**WEIGHT**: [" << i << "] [" << j << "] " << "[" << k << "] = " << w << endl;
				k++;
			}
			k = 0;
			j++;
		}
		j = 0;
		i++;
	}













}
/*
* 
* two things either make changes to the end neuron because its acting like a layer 
*
*/

