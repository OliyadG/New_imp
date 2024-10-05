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
	

	//b_Bais = dist(random_engine) * 0.5;
	b_Bais = 0; //change
}
class Network {

public:
	Network(int numberOfInputLayerNeuron, int numberOfHiddenLayers, int numberOfNeurons, int numberOfOutputNeurons);
	void feedForward();
	double getWeightedSum(int layerIndex, int NueronIndex);
	void feedForwardFlexableOutputs(int WhichActivationFunc = SIGMOID);
	void feedForwardFlexableInputs();



	//activations

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
	void costFunction(int WhichCostFunction);
	void derivativeOfCostFunction(int WhichCostFunction);

	void MSEcostFunction();
	void derivativeOfMSEcostFunction();

	void BinaryCrossEntropyLoss();
	void DerivativeOfBinaryCrossEntropyLoss();

	void ClipedBinaryCrossEntropyLoss();
	void DerivativeOfClipedBinaryCrossEntropyLoss();

	void backPropagationCalculateGradient(int indexOfLayer, int indexOfneuron);
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



	//static networksizes


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

	//initializations

	networkLayerSize = 1 + numberOfHiddenLayers + 1;
	inputLayersize = numberOfInputLayersNeuron;
	hiddenLayersize = numberOfHiddenNeurons;
	outputLayerSize = numberOfOutputNuerons;

	cost = 0.0;
	etaLearningRate = 0.0;

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
	double tempWeightedSum = 0.0;

	//i'll take each weight vector of each layer's indexed neuron and do the weightedsum and return it.
	// cout << "called for layer: " << layerIndex << "- and for neuronIndex: " << neuronIndex<<endl;
	for (auto& nueron : Layers[layerIndex])
	{
		double weight = nueron.getweightsAndConnections()[neuronIndex];
		double x = nueron.getInput();
		double b = nueron.getBais();

		tempWeightedSum = tempWeightedSum + (weight * x + b);
		//cout << "\nweight = " <<weight<<" x="<<x<<" b="<<b<<"--\n";
	}

	return tempWeightedSum;
}


//done
























void Network::feedForwardFlexableOutputs(int WhichActivationFunc)
{
	double weightedSum;

	int singleLayer = Layers.size() - 2; //-2 because its coming from hiddenl layer
	for (int singleNeuron = 0; singleNeuron < Layers[singleLayer + 1].size(); singleNeuron++)//for every output
	{

		weightedSum = getWeightedSum(singleLayer, singleNeuron); //sends the layer and the index number of the next nueron for the weight to be extracted for
		Layers[singleLayer + 1][singleNeuron].setWeightedSum(weightedSum);
		double Activated;



		switch (WhichActivationFunc)
		{
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
			double alpha = 0.1;
			Activated = leaky_reluActivationFunction(weightedSum, alpha);

		case SWISH:
			Activated = swishActivationFunction(weightedSum);
			break;

		case SOFTPLUS:
			Activated =softplus(weightedSum);
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
			Activated =  softmax(weightedSum);
			break;

		default:
			cerr << "***Activation Function Jumped!!**\n\a";
			break;
		}
		
		Output.push_back(Activated);
	}
}
void Network::feedForwardFlexableInputs()
{
	int size = Layers.front().size();
	auto& inputNeuron = Layers.front();

	for (int firstLayerNeurons = 0; firstLayerNeurons < size; firstLayerNeurons++)
	{
		inputNeuron[firstLayerNeurons].set_xInput(Input[firstLayerNeurons]);
		//calculate weighted sum, the weighted sum is the input its self.
		inputNeuron[firstLayerNeurons].setWeightedSum(Input[firstLayerNeurons]);
	}

}
void Network::feedForward()
{
	//intput // refract here so that its flexable
	feedForwardFlexableInputs();

	double weightedSum;

	for (int singleLayer = 0; singleLayer < Layers.size() - 1; singleLayer++)//changed to -2
	{
		
		
		
		
		if (singleLayer == Layers.size() - 2)
		{
			for (int singleNeuron = 0; singleNeuron < Layers[singleLayer + 1].size(); singleNeuron++)
			{
				/*
				* for every layer sent, a neuron of nextlayer is given the weighted sum of the current layers neurons, repectivly indexed
				* what i want to make sure now is the last 2, output neurons have thereweighted sum activated valueas an out put
				* so what we gonna do is calculate the weighted sum without feeding forward. so as to avoid forward indexing
				* so now i understand that the output, the activated weightedsum is also needed here at the output neurons
				*/
				weightedSum = getWeightedSum(singleLayer, singleNeuron); //sends the layer and the index number of the next nueron for the weight to be extracted for
				Layers[singleLayer + 1][singleNeuron].setWeightedSum(weightedSum);
				double Activated;
				Activated = sigmoidActivationFucntion(weightedSum);
				Output.push_back(Activated);
			}
			break;
		}
		
		else {
			for (int singleNeuron = 0; singleNeuron < Layers[singleLayer + 1].size(); singleNeuron++)
			{
				/*
				* for every layer sent, a neuron of nextlayer is given the weighted sum of the current layers neurons, repectivly indexed
				* what i want to make sure now is the last 2, output neurons have thereweighted sum activated valueas an out put
				* so what we gonna do is calculate the weighted sum without feeding forward. so as to avoid forward indexing
				* so now i understand that the output, the activated weightedsum is also needed here at the output neurons
				*/

				weightedSum = getWeightedSum(singleLayer, singleNeuron); //sends the layer and the index number of the next nueron for the weight to be extracted for
				Layers[singleLayer + 1][singleNeuron].setWeightedSum(weightedSum);
				




				double Activated = tanhActivationFunction(weightedSum);
				//double Activated = r
				//cout <<"\n" << Layers[singleLayer + 1][singleNeuron].getInput() << "******************************** start" << endl;
				Layers[singleLayer + 1][singleNeuron].set_xInput(Activated);
				//cout<<"\n" << Layers[singleLayer + 1][singleNeuron].getInput() << "****************************** end" << endl;

				//cout << "Weightedsum: =" << weightedSum << endl << "Activated: = " << Activated << endl << endl;
				//cout << "\ninputs weightedsum: " << Layers[0][0].getWeightedSum() << "second: weightedsum" << Layers[0][1].getWeightedSum()<<endl;

			}
		}
	}


	//experiment change for the last layer






	//output
	//feedForwardFlexableOutputs(TANH);/// remove
	//feedForwardFlexableOutputs(SIGMOID);//change

}
void Network::costFunction(int WhichCostFunction)
{
	switch (WhichCostFunction)
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
		
	default:
		break;
	}
}
void Network::derivativeOfCostFunction(int WhichCostFunction)
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
	default:
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
		//cout << "\n*******************desiredoutput = " << desiredOutput[sameIndex] << " and output<<" << singleOutputNeurons + 1e-10 << endl;
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
void Network::backPropagationCalculateGradient(int indexOfLayer, int indexOfOutputNeurons)
{


	
	int indexOfFrontNeuron;
	int indexOfBackNeuron;


    // gradient for last layer;
	int indexOfLastNeuron = 0;
	for (auto deltacost : deltaCosts)
	{
		//double w = Layers[indexOfLayer][indexOfLastNeuron].getweightsAndConnections()[0];
		double x = Layers[indexOfLayer][indexOfLastNeuron].getWeightedSum();
		//double b = Layers[indexOfLayer][indexOfLastNeuron].getBais();

		//double weightedSumOfTheOutput = w * x + b;
		//chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivativeOfSigmoidFucntion(w * x + b));     //temporary chain
		chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivativeOfSigmoidFucntion(x));// chaged chsin optimize, take to deltacost and put it there

		//these 3 were for the last incorrect layer changed.
		//calculatedGradient[indexOfLayer][indexOfLastNeuron][0] = chainedStored[indexOfLayer][indexOfLastNeuron] * x;
		//chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivativeOfSigmoidFucntion(w * x + b) * w * derivateOfTanhActivationFunction(x)); // chained stored
		//chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivateOfTanhActivationFunction(w * x + b) * w * derivateOfTanhActivationFunction(x)); // remove

		indexOfLastNeuron++;
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
				calculatedGradient[currentLayerIndex - 1][indexOfBackNeuron][indexOfFrontNeuron] = chainedStored[currentLayerIndex][indexOfFrontNeuron]/*derivative of thanh(x)*/ * backLayerNeuron.getWeightedSum()/* x of current*/; //optimize backweight is there
				tempChained += (chainedStored[currentLayerIndex][indexOfFrontNeuron] * backWieght[indexOfFrontNeuron] * derivateOfTanhActivationFunction(backWeightedSum));
				
				

			
			}
			chainedStored[currentLayerIndex - 1][indexOfBackNeuron] = tempChained;
			

		}


	}


}
void Network::backPropagationPropagate()
{
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
				if (desiredOutput[0] == 0.999)
				{
					Layerweights[numberOfWeight] -= calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight]/10;
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

	pairedData.first = tempDataInput;
	pairedData.second = tempDataDesire;

	return pairedData;
}
int main()
{
	auto start = chrono::high_resolution_clock::now();//-------------------------------------------------------------------time cost calculation
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	//cout << "Time taken: " << duration.count() << " microseconds" << endl;





	//network creation
	Network tempNet = Network(2, 6, 7, 1);













	
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

	int epoch = 1001;

train:

	auto pairedInputAndDesire = trainingGen(epoch);











	//test for right changes
	vector<vector<double>> input = pairedInputAndDesire.first;
	vector<double> desire = pairedInputAndDesire.second;

	double tempinput = 0.0;
	int count = 0;
	while (count < epoch)
	{

		//separate learning
		if (count == epoch - 2)
		{
			int temp;
			cin >> temp;

			count = 0;

			pairedInputAndDesire = trainingGen(temp);
			epoch = temp;

			//test for right changes
			input = pairedInputAndDesire.first;
			desire = pairedInputAndDesire.second;

		}


		tempNet.setInput(input[count]);




		tempNet.setDesiredOutput(vector<double>{desire[count]});




		//step #2 feedforward.//-------------------------------------------------------passed!
		tempNet.feedForward();

		//step #3 calculate cost   //--------------------------------------------------passed!
		tempNet.costFunction(BCEL);



		//step #4 calculate cost/output

		tempNet.derivativeOfCostFunction(BCEL);

		//steo #5 do a gradientcalculation
		tempNet.backPropagationCalculateGradient(7, 0);
		//tempNet.SecondBackPropagationCalculateGradient(5, 0);//remove
		//step #6 do a back prop
		tempNet.backPropagationPropagate();

		// cout << "input 1: " << tempNet.getInput()[0]<<endl;
		 //cout << "input 2: " << tempNet.getInput()[1]<<endl;





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

		if (epoch == 50)
		{
			epoch = 500;
			break;
		}
	}

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
		if (ask == 901) { goto train; }
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

