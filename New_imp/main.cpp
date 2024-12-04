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

void outputVerbose(vector<double>& input, vector<double> desired, vector<double>& output, double cost);





class Neuron {
public:
	Neuron() {};

	void makeConnections(int numberOfNextLayerNeurons);


	// temp gets
	vector<double>& getweightsAndConnections() { return w; }
	double getInput() { return x_ActivatedInput; }
	double& getBais() { return b_Bais; }
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


	b_Bais = dist(random_engine)*0.1;

	//b_Bais < 0 ? b_Bais = 0 : b_Bais = 1; // leave it be. here dont forget
	//b_Bais = 0; //change
}
class Network {

public:
	//inits
	Network(int numberOfInputLayerNeuron, int numberOfHiddenLayers, int numberOfNeurons, int numberOfOutputNeurons, int hiddenLayerActivation, int outputLayerActivation, int costFunction);
	void allocGradientAndChained();

	//main methods
	void feedForward();
	void feedForwardFlexableOutputs(int WhichActivationFunc = SIGMOID);
	void feedForwardFlexableInputs();

	

	//activations
	double ActivationFunctionsChoice(int Choice, double weightedSum);
	double derivativeOfActivationFunction(int Choice, double weightedSum);

	double tanhActivationFunction(double weightedSum) { return tanh(weightedSum); }
	double derivateOfTanhActivationFunction(double weightedSum) { return 1 - (tanh(weightedSum) * tanh(weightedSum)); }

	double sigmoidActivationFucntion(double weightedSum) { return 1.0 / (1.0 + exp(-weightedSum)); }
	double derivativeOfSigmoidFucntion(double weightedSum) { double sigmoid = sigmoidActivationFucntion(weightedSum);	return sigmoid * (1.0 - sigmoid); }

	double reluActivationFunction(double x) { return (x > 0) ? x : 0; }
	double reluActivationFunction_derivative(double x) { return (x > 0) ? 1 : 0; }

	double leaky_reluActivationFunction(double x, double alpha) { return (x > 0) ? x : alpha * x; }
	double leaky_reluActivationFunction_derivative(double x, double alpha) { return (x > 0) ? 1 : alpha; }

	double swishActivationFunction(double x) { return x / (1 + exp(-x)); }
	double swishActivationFunction_derivative(double x) { double exp_x = exp(-x); double denominator = (1 + exp_x) * (1 + exp_x); return (exp_x * (x + 1) - 1) / denominator; }

	void softmax(double maxOuput);
	double softmax_derivative(double x) { double exp_x = exp(x);	double denominator = (1 + exp_x) * (1 + exp_x);		return exp_x / denominator; }

	double softplus(double x) { return log(1 + exp(x)); }
	double softplus_derivative(double x) { return 1 / (1 + exp(-x)); }

	double softsign(double x) { return x / (1 + abs(x)); }
	double softsign_derivative(double x) { double denominator = (1 + abs(x)) * (1 + abs(x));		return 1 / denominator; }

	double bent_identity(double x) { return (sqrt(x * x + 1) - 1) / 2 + x; }
	double bent_identity_derivative(double x) { return x / sqrt(x * x + 1) + 1; }

	double hardtanh(double x) { if (x < -1) return -1;		if (x > 1) return 1;		return x; }
	double hardtanh_derivative(double x) { if (x < -1 || x > 1) return 0;		return 1; }

	double selu(double x) { if (x < 0) return 1.0507 * exp(x) - 1.0507;	return 1.0507 * x; }
	double selu_derivative(double x) { if (x < 0) return 1.0507 * exp(x);		return 1.0507; }

	double elu(double x) { if (x < 0) return exp(x) - 1;		return x; }
	double elu_derivative(double x) { if (x < 0) return exp(x);		return 1; }





	//costs
	void costFunctionChoice();
	void derivativeOfCostFunctionChoice();

	void MSEcostFunction();
	void derivativeOfMSEcostFunction();

	void BinaryCrossEntropyLoss();
	void DerivativeOfBinaryCrossEntropyLoss();

	void ClipedBinaryCrossEntropyLoss();
	void DerivativeOfClipedBinaryCrossEntropyLoss();

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



	//gets
	vector<vector<Neuron>>& getLayers(void) { return Layers; }
	vector<double>& getOutput() { return Output; }
	vector<double>& getInput() { return Input; }
	vector<double>& getDeltacosts() { return deltaCosts; }
	vector<vector<vector<double>>>& getCalculatedGradient() { return calculatedGradientWeight; }
	double getWeightedSum(int layerIndex, int NueronIndex);
	double getCost() const { return cost; }

	//set
	void setInput(vector<double> input) { Input = input; }
	void setOutput(vector<double> output) { Output = output; }
	void setDesiredOutput(vector<double> desire) { desiredOutput = desire; }
	void setHiddenLayerActivation(int function){ hiddenLayerActivation = function;}
	void setOutputLayerActivation(int function){ outputLayerActivation = function;}
	void setCostFunction(int function){ costFunction = function;}
	void setCatagoricalSoftMaxCond(bool cond) { catagorical_softmax = cond; }



	//reset
	void resetOutput() { Output.clear(); }
	void resetInput() { Input.clear(); }
	void resetDesired() { desiredOutput.clear(); }
	void resetCostsDeltas() { deltaCosts.clear(); cost = 0.0; }
	void resetGradient() { calculatedGradientWeight.clear(); }
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



	vector<vector<vector<double>>> calculatedGradientWeight;
	vector<vector<double>> calculatedGradientBias;
	vector < vector <double>> chainedStored;
	vector<vector<Neuron>> Layers;
	vector<double> Input;
	vector<double> Output;
	vector<double> desiredOutput;

	double etaLearningRate = 0.3;

	double cost;
	vector<double> deltaCosts;

	int hiddenLayerActivation;
	int outputLayerActivation;
	int costFunction;



	//static variables ADJUSTABLE!! here
	// 
		//for huber
	double delta = 0.5;//tolerance adjust as needed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
	//for focal
	double alpha = 0.5, gamma = 0.5; // adjust as needed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	//leaky
	double alphaLeaky = 0.01;


	//catagorical settings

	bool catagorical_softmax = false;


};
void Network::allocGradientAndChained()
{

	calculatedGradientWeight = vector<vector<vector<double>>>(networkLayerSize - 1);//gradient allocation
	calculatedGradientBias = vector<vector<double>>(networkLayerSize);//
	chainedStored = vector<vector<double>>(networkLayerSize); // chainedstore allocatio

	// Allocate first layer
	calculatedGradientWeight[0].resize(inputLayersize, vector<double>(hiddenLayersize));
	chainedStored[0].resize(inputLayersize);
	calculatedGradientBias[0].resize(inputLayersize);
	// Allocate hidden layers
	for (int i = 1; i < networkLayerSize - 1; i++) {
		if (i == networkLayerSize - 2) {
			calculatedGradientWeight[i].resize(hiddenLayersize, vector<double>(outputLayerSize));
			chainedStored[i].resize(hiddenLayersize);
			calculatedGradientBias[i].resize(hiddenLayersize);
		}
		else {
			calculatedGradientWeight[i].resize(hiddenLayersize, vector<double>(hiddenLayersize));
			calculatedGradientBias[i].resize(hiddenLayersize);
			chainedStored[i].resize(hiddenLayersize);
		}
	}
	// Allocate last layer
	//calculatedGradientWeight[networkLayerSize - 1].resize(outputLayerSize, vector<double>(1)); changed
	chainedStored[networkLayerSize - 1].resize(outputLayerSize);
}
Network::Network(int numberOfInputLayersNeuron, int numberOfHiddenLayers, int numberOfHiddenNeurons, int numberOfOutputNuerons, int hiddenLayerActivation, int outputLayerActivation, int costFunction)
{

	/*
	* set -hiddenActive, outputActive,costFunc
	*/

	setHiddenLayerActivation(hiddenLayerActivation);
	setOutputLayerActivation(outputLayerActivation);
	setCostFunction(costFunction);


	/*
	* initialization
	*/

	networkLayerSize = 1 + numberOfHiddenLayers + 1;
	inputLayersize = numberOfInputLayersNeuron;
	hiddenLayersize = numberOfHiddenNeurons;
	outputLayerSize = numberOfOutputNuerons;

	cost = 0.0;


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
			
			if (countLayers != numberOfHiddenLayers) {						// the last layer before the output layer should only make as many output neurons are in the outpputlayer
				tempNeuron.makeConnections(numberOfHiddenNeurons);
			}
			else {
				tempNeuron.makeConnections(numberOfOutputNuerons);
			}
			
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
	switch (Choice) {

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
	double maxOutput;//softmax's argument
	for (int singleNeuron = 0; singleNeuron < Layers[singleLayer + 1].size(); singleNeuron++)//for every output
	{



		weightedSum = getWeightedSum(singleLayer, singleNeuron); //sends the layer and the index number of the next nueron for the weight to be extracted for
		//we have weightedsum = (w1x+b * w2x+b....)
		Layers[singleLayer + 1][singleNeuron].setWeightedSum(weightedSum);
		
		//optionally - change
		double Activated = ActivationFunctionsChoice(functionChoice, weightedSum); //change softmax
		
		Layers[singleLayer + 1][singleNeuron].set_xInput(Activated);
		//optionally
		/*the risky implimentation
		* i'm not going to push the value into output
		* but i will push the weighted sum as usual
		* 
		* crucial thing here is am going to find the max value that is going to
		* be passed to the softmax
		* 
		*/

		
		
		//Output.push_back(Activated); 
		if (catagorical_softmax)
		{
			maxOutput = max(maxOutput, weightedSum);//get maxoutput for softmax
			Output.push_back(weightedSum); //for softmax change
		}
		else {
			Output.push_back(Activated);
		}
		
	}

	/*
	* if there it's catagorical then softmax is applied here
	* check if the condition catagorical_softmax is true
	* if then apply softmax for all outputs respectivly
	*/

	if (catagorical_softmax)
	{
		softmax(maxOutput);
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

			double Activated = ActivationFunctionsChoice(hiddenLayerActivation, weightedSum);

			Layers[singleLayer + 1][singleNeuron].set_xInput(Activated);

		}

	}

	//output with choiceActivationFunc
	feedForwardFlexableOutputs(outputLayerActivation);

}
void Network::costFunctionChoice()
{


	switch (costFunction)
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
void Network::derivativeOfCostFunctionChoice()
{

	switch (costFunction)
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


//costs
void Network::BinaryCrossEntropyLoss()
{
	int syncIndex = 0;
	cost = 0;


	for (auto& singleOutputNeurons : Output)
	{

		cost -= (desiredOutput[syncIndex] * log(singleOutputNeurons) + ((1 - desiredOutput[syncIndex]) * log(1 - singleOutputNeurons)));

		syncIndex++;
	}
	cost /= Output.size();

}
void Network::DerivativeOfBinaryCrossEntropyLoss()
{
	int syncIndex = 0;
	deltaCosts.clear();
	for (auto& singleOutputNeurons : Output)
	{
		deltaCosts.push_back(-(desiredOutput[syncIndex] / singleOutputNeurons + 1e-10) - ((1 - desiredOutput[syncIndex]) / ((1 - singleOutputNeurons) + 1e-10)));
		syncIndex++;
	}
}
void Network::MSEcostFunction()
{
	int syncIndex = 0;
	cost = 0;
	for (auto& singleOutputNeurons : Output)
	{
		cost += pow(singleOutputNeurons - desiredOutput[syncIndex], 2);
		syncIndex++;
	}
	cost /= Output.size();

}
void Network::derivativeOfMSEcostFunction()
{
	int syncIndex = 0;
	deltaCosts.resize(Output.size());
	for (auto& singleOutputNeurons : Output)
	{
		deltaCosts[syncIndex] = 2 * (singleOutputNeurons - desiredOutput[syncIndex]);
		syncIndex++;
	}
}
void Network::ClipedBinaryCrossEntropyLoss()
{
	int syncIndex = 0;
	cost = 0;
	for (auto& singleOutputNeurons : Output)
	{
		double clipedOutSafeValue = max(min(singleOutputNeurons, 1 - 1e-10), 1e-10);
		cost -= (desiredOutput[syncIndex] * log(clipedOutSafeValue) + (1 - desiredOutput[syncIndex] * log(1 - clipedOutSafeValue)));
		syncIndex++;
	}
	cost /= Output.size();
}
void Network::DerivativeOfClipedBinaryCrossEntropyLoss()
{
	int syncIndex = 0;
	deltaCosts.resize(Output.size());
	for (auto& singleOutputNeurons : Output)
	{
		// Clipping the output to avoid log(0) or division by zero
		double clipedOutSafeValue = max(min(singleOutputNeurons, 1 - 1e-10), 1e-10);

		// Calculate the gradient of the cost w.r.t each output neuron
		deltaCosts[syncIndex] = -(desiredOutput[syncIndex] / clipedOutSafeValue) +
			((1 - desiredOutput[syncIndex]) / (1 - clipedOutSafeValue));

		syncIndex++;
	}
}





void Network::CategoricalCrossEntropyLoss() {
	int syncIndex = 0;
	cost = 0;


	for (auto& singleOutputNeurons : Output)
	{
		cost -= desiredOutput[syncIndex] * log(singleOutputNeurons + 1e-10);
		syncIndex++;
	}
	cost /= Output.size();
}
void Network::DerivativeOfCategoricalCrossEntropyLoss() {
	int syncIndex = 0;
	deltaCosts.clear();
	for (auto& singleOutputNeurons : Output)
	{
		deltaCosts.push_back(singleOutputNeurons - desiredOutput[syncIndex]);	//simplified -desiredOutput[i] / softmaxedOutput[i] * softmaxedOutput[i] * (1 - softmaxedOutput[i])
		/* just brilliant
		https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/#cross-entropy-loss
		*/
		syncIndex++;
	}
}
void Network::softmax(double maxOutput)
{
	/*
	* Compute the maximum value in the output to improve numerical stability
	* Compute the sum of exponentials of (output - maxOutput)
	* Compute softmax values

	*/


	//stability, max val
	//double maxOutput = *max_element(Output.begin(), Output.end()); //change


	cout << "--------" << maxOutput << endl;
	for (auto ss : Output)
	{
		cout << ss << endl;
	}

	// sum exponetial
	double expSum = 0.0;
	for (auto& singleOutput : Output) {
		expSum += exp(singleOutput - maxOutput); //stability
	}

	// softmax
	for (auto& singleOutput : Output) {
		singleOutput = exp(singleOutput - maxOutput) / expSum;
	}

	cout << "--------" << maxOutput << endl;
	for (auto ss : Output)
	{
		cout << ss << endl;
	}
}








void Network::HuberLoss(double delta) {

	int syncIndex = 0;

	cost = 0;
	for (auto& singleOutputNeurons : Output) {
		double error = singleOutputNeurons - desiredOutput[syncIndex];
		if (abs(error) <= delta) {
			cost += 0.5 * error * error;
		}
		else {
			cost += delta * (abs(error) - 0.5 * delta);
		}
		syncIndex++;
	}
	cost /= Output.size();
}
void Network::DerivativeOfHuberLoss(double delta) {
	int syncIndex = 0;
	deltaCosts.clear();
	for (auto& singleOutputNeurons : Output) {
		double error = singleOutputNeurons - desiredOutput[syncIndex];
		if (abs(error) <= delta) {
			deltaCosts.push_back(error);
		}
		else {
			deltaCosts.push_back(delta * sign(error));
		}
		syncIndex++;
	}
}
void Network::FocalLoss(double alpha, double gamma) {
	int syncIndex = 0;
	cost = 0;

	for (auto& singleOutputNeurons : Output) {
		double pt = singleOutputNeurons;
		cost -= alpha * pow(1 - pt, gamma) * desiredOutput[syncIndex] * log(pt + 1e-10) +
			(1 - alpha) * pow(pt, gamma) * (1 - desiredOutput[syncIndex]) * log(1 - pt + 1e-10);
		syncIndex++;
	}
	cost /= Output.size();
}
void Network::DerivativeOfFocalLoss(double alpha, double gamma) {
	int syncIndex = 0;
	deltaCosts.clear();
	for (auto& singleOutputNeurons : Output) {
		double pt = singleOutputNeurons;
		deltaCosts.push_back(alpha * pow(1 - pt, gamma) * desiredOutput[syncIndex] / (pt + 1e-10) -
			(1 - alpha) * pow(pt, gamma) * (1 - desiredOutput[syncIndex]) / (1 - pt + 1e-10));
		syncIndex++;
	}
}
void Network::DiceLoss() {
	int syncIndex = 0;
	double intersection = 0;
	double sum = 0;

	for (auto& singleOutputNeurons : Output) {
		intersection += singleOutputNeurons * desiredOutput[syncIndex];
		sum += pow(singleOutputNeurons, 2) + pow(desiredOutput[syncIndex], 2);
		syncIndex++;
	}
	cost = 1 - (2 * intersection) / (sum + 1e-10);
}
void Network::DerivativeOfDiceLoss() {
	int syncIndex = 0;
	deltaCosts.clear();
	double intersection = 0;
	double sum = 0;

	for (auto& singleOutputNeurons : Output) {
		intersection += singleOutputNeurons * desiredOutput[syncIndex];
		sum += pow(singleOutputNeurons, 2) + pow(desiredOutput[syncIndex], 2);
		syncIndex++;
	}

	for (auto& singleOutputNeurons : Output) {
		deltaCosts.push_back((2 * desiredOutput[syncIndex] * (sum - 2 * intersection)) / (sum * sum + 1e-10));
	}
}


void Network::backPropagationCalculateGradient() {
	int numLayers = Layers.size();

	for (int neuronIndex = 0; neuronIndex < deltaCosts.size(); ++neuronIndex) {
		double weightedSum = Layers[numLayers - 1][neuronIndex].getWeightedSum();
		if (catagorical_softmax) {
			chainedStored[numLayers - 1][neuronIndex] = deltaCosts[neuronIndex]; //change dont forget
		}
		else {
			chainedStored[numLayers - 1][neuronIndex] = deltaCosts[neuronIndex] *
				derivativeOfActivationFunction(outputLayerActivation, weightedSum);
		}
	}
	for (int layerIndex = numLayers - 2; layerIndex >= 0; --layerIndex) {
		for (int backNeuronIndex = 0; backNeuronIndex < Layers[layerIndex].size(); ++backNeuronIndex) {
			Neuron& backNeuron = Layers[layerIndex][backNeuronIndex];
			double backActivation = backNeuron.getInput();
			double backWeightedSum = backNeuron.getWeightedSum();
			double chainSum = 0.0;

			for (int frontNeuronIndex = 0; frontNeuronIndex < Layers[layerIndex + 1].size(); ++frontNeuronIndex) {
				Neuron& frontNeuron = Layers[layerIndex + 1][frontNeuronIndex];
				double weight = backNeuron.getweightsAndConnections()[frontNeuronIndex];
				double frontChain = chainedStored[layerIndex + 1][frontNeuronIndex];//leave me alone

				// Gradient with respect to weights
				calculatedGradientWeight[layerIndex][backNeuronIndex][frontNeuronIndex] =
					frontChain * backActivation;

				// Accumulate chained gradient for the back neuron
				chainSum += frontChain * weight;
			}

			// Store chained gradient for the back neuron
			chainedStored[layerIndex][backNeuronIndex] = chainSum *
				derivativeOfActivationFunction(hiddenLayerActivation, backWeightedSum);
		}
	}
}


//done



























void Network::backPropagationPropagate() {
	int numLayers = calculatedGradientWeight.size();

	for (int layerIndex = 0; layerIndex < numLayers; layerIndex++) {
		int numNeurons = calculatedGradientWeight[layerIndex].size();

		for (int neuronIndex = 0; neuronIndex < numNeurons; neuronIndex++) {
			auto& weights = Layers[layerIndex][neuronIndex].getweightsAndConnections();
			auto& bias = Layers[layerIndex][neuronIndex].getBais();

			// Update bias
			bias -= etaLearningRate * calculatedGradientBias[layerIndex][neuronIndex];

			int numWeights = weights.size();
			for (int weightIndex = 0; weightIndex < numWeights; weightIndex++) {
				// Update weights
				weights[weightIndex] -= etaLearningRate * calculatedGradientWeight[layerIndex][neuronIndex][weightIndex];
			}
		}
	}
}


pair<vector<vector<double>>, vector<vector<double>>> trainingGenLogicalOperations(int size = 1000)
{
	pair<vector<vector<double>>, vector<vector<double>>> pairedData;
	vector<vector<double>> stempDataDesire;
	vector<vector<double>> tempDataDesire;
	vector<vector<double>> tempDataInput;
	for (int a = 0; a < size; ++a) {
		
		tempDataInput.push_back({ static_cast<double>(0), static_cast<double>(0) });
		//tempDataDesire.push_back({ static_cast<double>(0)});
		tempDataDesire.push_back({ static_cast<double>(0), static_cast<double>(1) });

		tempDataInput.push_back({ static_cast<double>(0), static_cast<double>(1) });
		//tempDataDesire.push_back({ static_cast<double>(1) });
		tempDataDesire.push_back({ static_cast<double>(1), static_cast<double>(0) });

		tempDataInput.push_back({ static_cast<double>(1), static_cast<double>(1) });
		//tempDataDesire.push_back({ static_cast<double>(0) });
		tempDataDesire.push_back({ static_cast<double>(0), static_cast<double>(1) });

		tempDataInput.push_back({ static_cast<double>(1), static_cast<double>(0) });
		//tempDataDesire.push_back({ static_cast<double>(1) });
		tempDataDesire.push_back({ static_cast<double>(1), static_cast<double>(0) });
	}
	pairedData.first = tempDataInput;
	pairedData.second = tempDataDesire;
	return pairedData;
}

pair<vector<vector<double>>, vector<vector<double>>> trainingGenLogicalMathFunction(int size = 1000)
{
	pair<vector<vector<double>>, vector<vector<double>>> pairedData;
	vector<vector<double>> tempDataDesire;
	vector<vector<double>> tempDataInput;
	for (int a = 0; a < size; ++a) {

		tempDataInput.push_back({ static_cast<double>(a) });
		tempDataDesire.push_back({ static_cast<double>(a*a) });
	}
	pairedData.first = tempDataInput;
	pairedData.second = tempDataDesire;
	return pairedData;
}



/*
* 
* 
* **********************where am i******
* 
* 
* i have chaanges active function to whole TANH,
* im also using MSE as a cost function
* and i'm also adjusting the gradient of the bias
* and its working very well
* and i have also no need of using the condition to use - + for gradient
* 
* 
* this network is working smothly and brilliantly




*/











/*
* why can't it go up more than 1?

*/



void checkWeightsAndBiases(vector<vector<Neuron>>& Layers)
{
	cout << "\nweights and biases****\n";

	int sizeLayer = Layers.size();
	for (int numberOfLayer = 0; numberOfLayer < sizeLayer; numberOfLayer++)
	{
		cout << "layer**: " << numberOfLayer << endl;
		int sizeNeuron = Layers[numberOfLayer].size();
		for (int numberOfNeuron = 0; numberOfNeuron < sizeNeuron; numberOfNeuron++)
		{
			auto& bias = Layers[numberOfLayer][numberOfNeuron].getBais();
			auto& weights = Layers[numberOfLayer][numberOfNeuron].getweightsAndConnections();
			int sizeWeight = weights.size();
			cout << "weights**" << endl;
			for (int numberOfweight = 0; numberOfweight < sizeWeight; numberOfweight++)
			{
				//prints all weight
				cout << "L[" << numberOfLayer << "]" << "N[" << numberOfNeuron << "]" << " W = " << weights[numberOfweight] << endl;
			}
			cout << "\n+ bias**\n" << "L[" << numberOfLayer << "]" << "N[" << numberOfNeuron << "]" << " B= " << bias<<endl;
		}
	}


}

void train(int epoch, int datasetSize, Network& model, vector<vector<double>>& input, vector<vector<double>>& desire)
{
		vector<double>& output = model.getOutput();
		int count = 0;
		int syncIndex = 0;
		int syncIndexOutput = 0;
		int repeat = epoch / datasetSize;

		while (repeat > 0)
		{
			checkWeightsAndBiases(model.getLayers());//check weights
			while (count < epoch)
			{									/******START SINGLE TRAING******/
			//separate learning
			
				
				
			    
				model.setInput(input[count]);
				model.setDesiredOutput(desire[count]);
				//step #2 feedforward.
				model.feedForward();
				//step #3 calculate cost  
				model.costFunctionChoice();
				//step #4 calculate cost/output
				model.derivativeOfCostFunctionChoice();
				//steo #5 do a gradientcalculation
				model.backPropagationCalculateGradient();
				//step #6 do a back prop
				model.backPropagationPropagate();
				/******END SINGLE TRAING******/

//output input

				

				outputVerbose(input[count], desire[count], output, model.getCost());


				//for (auto& inp : input[count])
				//{
				//	cout << "input [" << syncIndex << "]: " << inp;
				//	cout << endl;
				//	syncIndex++;
				//}
				////output desired and actual
				//for (auto& dsi : desire[count])
				//{
				//	cout << "desired [" << syncIndexOutput << "]: " << dsi << " ||| output [" << syncIndexOutput << "]: " << output[syncIndexOutput];
				//	cout << endl;
				//	syncIndexOutput++;
				//}
				////cost
				//cout << "***** COST ****: " << model.getCost() << endl;


				model.resetOutput();
				model.resetDesired();
				model.resetInput();
				model.resetCostsDeltas();
				model.resetChained();
				model.resetGradient();
				model.allocGradientAndChained();
				count++;
				syncIndex = 0;
				syncIndexOutput = 0;

			}
			repeat--;
			count = 0;
			checkWeightsAndBiases(model.getLayers());//check weights
		}
}

void testModel(Network& model, vector<double> inpt)
{

		model.setInput(inpt);
		model.feedForward();
		auto& output = model.getOutput();

		for (auto singles : output)
		{
			cout << singles << endl;
		}
		model.resetOutput();
		model.resetDesired();
		model.resetInput();
}

void outputVerbose(vector<double>& input, vector<double> desired, vector<double>& output, double cost)
{

	//show input
	cout << "\n*****input: \n";
	for (auto& in : input)
	{
		cout << "<<---- "<<in << endl;
	}
	//show output
	cout << "*****output: \n";
	for (auto& out : output)
	{
		cout <<"---->> " << out << endl;
	}
	//show output
	cout << "*****desired: \n";
	for (auto& des : desired)
	{
		cout << "===== " << des<< endl;
	}
	//show cost
	cout << "\n*******************cost : " << cost<<endl;

	


}





int main()
{
	//auto start = chrono::high_resolution_clock::now();//-------------------------------------------------------------------time cost calculation
	//auto stop = chrono::high_resolution_clock::now();
	//auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	//cout << "Time taken: " << duration.count() << " microseconds" << endl;

	//network creation
	auto model = Network(2, 1, 4, 2, RELU, RELU, CCE);//init net
	model.setCatagoricalSoftMaxCond(true);//catagorical!
	auto& Layers = model.getLayers();							//get Layers
	
	
	//get training data
	int epoch = 100000;												//number of trainigs - 1000 trained 50X
	int dataSet = 100000;											//number of dataset - single generated data

	
	//test for right changes
	//auto pairedInputAndDesire = trainingGenLogicalMathFunction(epoch); //x^2
	auto pairedInputAndDesire = trainingGenLogicalOperations(epoch);
	vector<vector<double>> input = pairedInputAndDesire.first;
	vector<vector<double>> desire = pairedInputAndDesire.second;
	//checkWeightsAndBiases(Layers);
	train(epoch, dataSet, model, input, desire);
	//checkWeightsAndBiases(Layers);

	testModel(model, vector<double>{0, 0});
	testModel(model, vector<double>{1, 1});
	testModel(model, vector<double>{1, 0});
	testModel(model, vector<double>{0, 1});


}


/*
* i have made changes to the catagorical cross entropy cost
* it now can very correctly handle classification.
* the only change made is the derivative of the cost
* anther thing is the softmax, the last layer where the output is filled.
* if the catagorical_softmax is set true
* then it applies the softmax
* then the other thing that bothered me was integrating the softmax derivative inside the cost derivative. 
* since it can be simplified and i would't have to modify the backpropgradient calculator
* 
* 
* 
* now what i need is to change:
* 1.create a supparate outputVerbose function make it flexable that i wouldn't have to change it when the network architecture changes
* 2.create a dataset maker or image to vector in this case.
* 3.test the network with a single set of picutre like the number 3
* 


*/