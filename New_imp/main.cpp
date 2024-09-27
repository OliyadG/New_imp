#include<iostream>
#include<vector>
#include<math.h>
#include<random>
#include <chrono>
using namespace std;

constexpr auto TANH = 0;;
constexpr auto SIGMOID = 1;
constexpr auto RELU = 2;

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

	b_Bais = dist(random_engine) * 0.1;
}

class Network {

public:
	Network(int numberOfInputLayerNeuron, int numberOfHiddenLayers, int numberOfNeurons, int numberOfOutputNeurons);
	void feedForward();
	double getWeightedSum(int layerIndex, int NueronIndex);
	void feedForwardFlexableOutputs(int WhichActivationFunc = SIGMOID);
	void feedForwardFlexableInputs();


	//activations
	double tanhActivationFunction(double weightedSum) { return tanh(weightedSum); };
	double derivateOfTanhActivationFunction(double weightedSum);
	

	double sigmoidActivationFucntion(double weightedSum); 
	double derivativeOfSigmoidFucntion( double weightedSum);

	double softMax();



	//costs
	void costFunction(int WhichCostFunction);
	void derivativeOfCostFunction(int WhichCostFunction);

	void MSEcostFunction();
	void derivativeOfMSEcostFunction();

	void BinaryCrossEntropyLoss();
	void DerivativeOfBinaryCrossEntropyLoss();

	void ClipedBinaryCrossEntropyLoss();
	void DerivativeOfClipedBinaryCrossEntropyLoss();

	void backPropagation(int indexOfLayer, int indexOfneuron, double store);
	


	//temp gets
	vector<vector<Neuron>> getLayers(void) { return Layers; }
	vector<double> getOutput() { return Output; }
	vector<double> getInput() { return Input; }
	double getCost() { return cost; }
	vector<double> getDeltacosts() { return deltaCosts; }

	// temp sets
	void setInput(vector<double> input) { Input = input; }
	void setOutput(vector<double> output) { Output = output; }
	void resetOutput() { Output.clear(); }
	void setDesiredOutput(vector<double> desire) { desiredOutput = desire; }






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

	double etaLearningRate;

	double cost;
	vector<double> deltaCosts;



	//static networksizes


};

Network::Network(int numberOfInputLayersNeuron, int numberOfHiddenLayers, int numberOfHiddenNeurons, int numberOfOutputNuerons)
{


	networkLayerSize = 1 + numberOfHiddenLayers + 1;
	inputLayersize = numberOfInputLayersNeuron;
	hiddenLayersize = numberOfHiddenNeurons;
	outputLayerSize = numberOfOutputNuerons;

	

	this->cost = 0.0;
	this->etaLearningRate = 0.0;

	//forchained stored allocations



	

	// if the network has a fixed sizes \/\/\/\/\/ would be enough
	// vector<vector<vector<double>>> calculatedGradient(maxLayerSize, vector<vector<double>>(maxNeuronSize, vector<double>(maxWeightSize)));


	auto start = std::chrono::high_resolution_clock::now();//-------------------------------------------------------------------time cost calculation

	calculatedGradient = vector<vector<vector<double>>>(networkLayerSize);//gradient allocation
	chainedStored = vector<vector<double>> (networkLayerSize); // chainedstore allocatio
	// Allocate first layer
	calculatedGradient[0].resize(numberOfInputLayersNeuron, vector<double>(numberOfHiddenNeurons));
	chainedStored[0].resize(numberOfInputLayersNeuron);
	// Allocate hidden layers
	for (int i = 1; i < networkLayerSize - 1; i++) {
		if (i == networkLayerSize - 2) {
			calculatedGradient[i].resize(numberOfHiddenNeurons, vector<double>(numberOfOutputNuerons));
			chainedStored[i].resize(numberOfHiddenNeurons);
		}
		else {
			calculatedGradient[i].resize(numberOfHiddenNeurons, vector<double>(numberOfHiddenNeurons));
			chainedStored[i].resize(numberOfHiddenNeurons);
		}
	}
	// Allocate last layer
	calculatedGradient[networkLayerSize - 1].resize(numberOfOutputNuerons, vector<double>(1));
	chainedStored[networkLayerSize - 1].resize(numberOfOutputNuerons);




	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "---Time taken: " << duration.count() << " microseconds" << std::endl;

	//---------------------------------------------------------------------------------------------------------------time cost







	
	Input = vector<double>(numberOfInputLayersNeuron);//-------------------------------------------------------------size of input





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
		tempNeuron.makeConnections(1);
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

void Network::feedForwardFlexableOutputs(int WhichActivationFunc)
{
	auto& lastLayer = Layers.back();
	
	switch (WhichActivationFunc)
	{
	case TANH:
			
			for (auto& outputNeurons : lastLayer)
			{
				//outputNeurons.setWeightedSum(outputNeurons.getweightsAndConnections()[0]/*w*/ * outputNeurons.getInput()/*x*/ + outputNeurons.getBais()/*b*/);
				//store outputs
				double weightedSum = outputNeurons.getweightsAndConnections()[0]/*w*/ * outputNeurons.getInput()/*x*/ + outputNeurons.getBais()/*b*/;
				Output.push_back(tanhActivationFunction(weightedSum));

				cout << "last layer weighted sums not output! : " << outputNeurons.getWeightedSum()<<endl<<"----\n";
			}
			break;

	case SIGMOID:
		
			for (auto& outputNeurons : lastLayer)
			{
				//outputNeurons.setWeightedSum(outputNeurons.getweightsAndConnections()[0]/*w*/ * outputNeurons.getInput()/*x*/ + outputNeurons.getBais()/*b*/);//i dont think this is needed here since the front feed had it calculated for us, weighted sum comes from that
				//store outputs
				double weightedSum = outputNeurons.getweightsAndConnections()[0]/*w*/ * outputNeurons.getInput()/*x*/ + outputNeurons.getBais()/*b*/;
				Output.push_back(sigmoidActivationFucntion(weightedSum));
				cout << "last layer weighted sums not output! : " << outputNeurons.getWeightedSum() << endl <<"bruh activated alew?: output "<< Output.back()<<endl<<" weightedsum output:"<<weightedSum << "----\n";
			}
			break;

	default:
		break;
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

double Network::derivateOfTanhActivationFunction(double weightedSum)
{
	 return 1 - (tanh(weightedSum) * tanh(weightedSum)); 
}

double Network::sigmoidActivationFucntion(double weightedSum)
{
	return 1.0 / (1.0 + exp(-weightedSum));
}

double Network::derivativeOfSigmoidFucntion(double weightedSum)
{
	double sigmoid = sigmoidActivationFucntion(weightedSum);

	return sigmoid * (1.0 - sigmoid);
}

void Network::feedForward()
{
	//intput // refract here so that its flexable
	feedForwardFlexableInputs();

	double weightedSum;

	for (int singleLayer = 0; singleLayer < Layers.size() - 1; singleLayer++)
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
			double Activated = tanhActivationFunction(weightedSum);

			//cout <<"\n" << Layers[singleLayer + 1][singleNeuron].getInput() << "******************************** start" << endl;
			Layers[singleLayer + 1][singleNeuron].set_xInput(Activated);
			//cout<<"\n" << Layers[singleLayer + 1][singleNeuron].getInput() << "****************************** end" << endl;

			//cout << "Weightedsum: =" << weightedSum << endl << "Activated: = " << Activated << endl << endl;
			//cout << "\ninputs weightedsum: " << Layers[0][0].getWeightedSum() << "second: weightedsum" << Layers[0][1].getWeightedSum()<<endl;
		}
	}


	//output
	feedForwardFlexableOutputs();

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
		BinaryCrossEntropyLoss();
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
		cost -= (desiredOutput[sameIndex] * log(singleOutputNeurons) + (1 - desiredOutput[sameIndex] * log(1 - singleOutputNeurons)));
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
		deltaCosts.push_back((desiredOutput[sameIndex] / singleOutputNeurons + 1e-10) - ((1 - desiredOutput[sameIndex]) / ((1 - singleOutputNeurons) + 1e-10)));
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

void Network::backPropagation(int indexOfLayer, int indexOfOutputNeurons, double store)
{


	
	
	vector < vector <double>> chainedStored;
	int indexOfFrontNeuron;
	int indexOfBackNeuron;






    // gradient for last layer;
	int indexOfLastNeuron = 0;
	for (auto deltacost : deltaCosts)
	{
		double w = Layers[indexOfLayer][indexOfLastNeuron].getweightsAndConnections()[0];
		double x = Layers[indexOfLayer][indexOfLastNeuron].getWeightedSum();
		double b = Layers[indexOfLayer][indexOfLastNeuron].getBais();

		double weightedSumOfTheOutput = w * x + b;
		chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivativeOfSigmoidFucntion(w * x + b));     //temporary chain
		calculatedGradient[indexOfLayer][indexOfLastNeuron][0] = chainedStored[indexOfLayer][indexOfLastNeuron] * x;
		chainedStored[indexOfLayer][indexOfLastNeuron] = (deltacost * derivativeOfSigmoidFucntion(w * x + b) * w * derivateOfTanhActivationFunction(x)); // chained stored
		
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
				calculatedGradient[currentLayerIndex - 1][indexOfBackNeuron][indexOfFrontNeuron] = chainedStored[currentLayerIndex][indexOfFrontNeuron]/*derivative of thanh(x)*/ * backLayerNeuron.getWeightedSum()/* x of current*/;
				tempChained += (chainedStored[currentLayerIndex][indexOfFrontNeuron] * backWieght[indexOfFrontNeuron] * derivateOfTanhActivationFunction(backWeightedSum));

			}
			chainedStored[currentLayerIndex - 1][indexOfBackNeuron] = tempChained;
			

		}


	}


}

int main()
{
	//auto start = std::chrono::high_resolution_clock::now();//-------------------------------------------------------------------time cost calculation
	// 
	//auto stop = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	//std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;


	Network tempNet = Network(2, 4, 4, 1);
	
	vector<vector<Neuron>> Layers = tempNet.getLayers();
	return 0;
	//step #1 set input
	tempNet.setInput(vector<double>{0.3, 0.6}); //-----------------------passed!

	//step #2 feedforward.
	tempNet.feedForward(); //--------------------------------------------passed!


	







/*
	vector<double> Input;
	vector<double> Output;
	vector<double> desiredOutput{vector<double>{1, 1}};





	Network tempNet = Network(2, 10, 10, 1);





	auto Layers = tempNet.getLayers();
	tempNet.setInput(vector<double>{0.3, 0.6});
	tempNet.setDesiredOutput(desiredOutput);

	while (true) {

		double x, y;
		cin >> x >> y;
		tempNet.setInput(vector<double>{x, y});
		tempNet.feedForward();

		cout << "outputNuron1: " << tempNet.getOutput()[0];
		cout << "\noutputNuron2: " << tempNet.getOutput()[1] << endl;
		

		tempNet.BinaryCrossEntropyLoss();
		tempNet.DerivativeOfBinaryCrossEntropyLoss();


		cout <<"cost " << tempNet.getCost()<<endl;
		//cout << "deltacost0 " << tempNet.getDeltacosts();
		tempNet.resetOutput();

		
	}
	*/
}

/*
* neural network with input, output and hiddenlayer made
* connection made with all neuron in the next layer
* weight given randomly
* b = bias
*weight= weight
* x = input/output value
* **********************
*
*
* missing, backpropagation! the delta, learning rate, and derivatevs.
*
* debug why out puts dont change even thought the inputs change
* be sure to read and impliment softmax, its incredible, if there are classifications probabilistic one, then their sum must equal 1. it does that! you must also impliment sigmoid
* 
* 
* 
* 
* changes!
* im going to make these changes
* im goig to make the output neuron 1
* just propability of 0-1
* sigmoid is going to be used.
* change the last layer to have a single output
* and make sure in feedforward to only get output of the single nueron
* and make changes to desired, output vectors because they are 2 sized
* and write the derivatives for all activations, it will be used later
* 
* an other far thing is, this is optional for now, during back prop its better to store the delta or value of the
* derivative of cost respect to out put, its only going to be used multiple times. so store it.
* then the derivative of the weighted sum is the tricky ones, as you propagate you will have to use the tanh derivative to solve that.
* you might be able to utilize recurrsion, or even matrixs...because at one go you might be able to calculate the derivative of 
* a layers neurons, because you cant go past without first adjesting!
* 



*/