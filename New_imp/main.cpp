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
	vector<double>& getVelocities()
	{
		return velocities;
	}


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
		//tempCon = dist(random_engine);
	}
	velocities.resize(numberOfNextLayerNeurons, 0.0);//remove

	b_Bais = dist(random_engine) * 0.1;
}

class Network {

public:
	Network(int numberOfInputLayerNeuron, int numberOfHiddenLayers, int numberOfNeurons, int numberOfOutputNeurons);
	void feedForward();
	double getWeightedSum(int layerIndex, int NueronIndex);
	void feedForwardFlexableOutputs(int WhichActivationFunc = SIGMOID);
	void feedForwardFlexableInputs();

	void SecondBackPropagationCalculateGradient(int indexOfLayer, int indexOfOutputNeurons);


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

	void backPropagationCalculateGradient(int indexOfLayer, int indexOfneuron);
	void backPropagationPropagate();
	


	//temp gets
	vector<vector<Neuron>> getLayers(void) { return Layers; }
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
	
	calculatedGradient = vector<vector<vector<double>>>(networkLayerSize);//gradient allocation
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
	calculatedGradient[networkLayerSize - 1].resize(outputLayerSize, vector<double>(1));
	chainedStored[networkLayerSize - 1].resize(outputLayerSize);
}

Network::Network(int numberOfInputLayersNeuron, int numberOfHiddenLayers, int numberOfHiddenNeurons, int numberOfOutputNuerons)
{

	//initializations

	networkLayerSize = 1 + numberOfHiddenLayers + 1;
	inputLayersize = numberOfInputLayersNeuron;
	hiddenLayersize = numberOfHiddenNeurons;
	outputLayerSize = numberOfOutputNuerons;

	//Input = vector<double>(numberOfInputLayersNeuron);//-------------------------------------------------------------size of input
	//Output = vector<double>(numberOfOutputNuerons); //---------------------------------------------------------------size of output
	//deltaCosts = vector<double>(numberOfOutputNuerons);//------------------------------------------------------------size of deltaCost
	cost = 0.0;
	etaLearningRate = 0.0;





	

	// if the network has a fixed sizes \/\/\/\/\/ would be enough
	// vector<vector<vector<double>>> calculatedGradient(maxLayerSize, vector<vector<double>>(maxNeuronSize, vector<double>(maxWeightSize)));


	//auto start = std::chrono::high_resolution_clock::now();//-------------------------------------------------------------------time cost calculation
	allocGradientAndChained();




	//auto stop = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	//std::cout << "---Time taken: " << duration.count() << " microseconds" << std::endl;

	//---------------------------------------------------------------------------------------------------------------time cost







	





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

				//cout << "last layer weighted sums not output! : " << outputNeurons.getWeightedSum()<<endl<<"----\n";
			}
			break;

	case SIGMOID:
		
			for (auto& outputNeurons : lastLayer)
			{
				//outputNeurons.setWeightedSum(outputNeurons.getweightsAndConnections()[0]/*w*/ * outputNeurons.getInput()/*x*/ + outputNeurons.getBais()/*b*/);//i dont think this is needed here since the front feed had it calculated for us, weighted sum comes from that
				//store outputs
				double weightedSum = outputNeurons.getweightsAndConnections()[0]/*w*/ * outputNeurons.getInput()/*x*/ + outputNeurons.getBais()/*b*/;
				Output.push_back(sigmoidActivationFucntion(weightedSum));
				//cout << "last layer weighted sums not output! : " << outputNeurons.getWeightedSum() << endl <<"bruh activated alew?: output "<< Output.back()<<endl<<" weightedsum output:"<<weightedSum << "----\n";
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
			//double Activated = r
			//cout <<"\n" << Layers[singleLayer + 1][singleNeuron].getInput() << "******************************** start" << endl;
			Layers[singleLayer + 1][singleNeuron].set_xInput(Activated);
			//cout<<"\n" << Layers[singleLayer + 1][singleNeuron].getInput() << "****************************** end" << endl;

			//cout << "Weightedsum: =" << weightedSum << endl << "Activated: = " << Activated << endl << endl;
			//cout << "\ninputs weightedsum: " << Layers[0][0].getWeightedSum() << "second: weightedsum" << Layers[0][1].getWeightedSum()<<endl;
		}
	}


	//output
	feedForwardFlexableOutputs(SIGMOID);

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
		cost -= (desiredOutput[sameIndex] * log(singleOutputNeurons) + (1 - desiredOutput[sameIndex]) * log(1 - singleOutputNeurons));
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


void Network::SecondBackPropagationCalculateGradient(int indexOfLayer, int indexOfOutputNeurons)//remove
{
	int indexOfFrontNeuron;
	int indexOfBackNeuron;
	double momentum = 0.9;   // Momentum coefficient
	double learningRate = 0.5; // Learning rate

	// Gradient for the last layer;
	int indexOfLastNeuron = 0;
	for (auto deltacost : deltaCosts)
	{
		Neuron& neuron = Layers[indexOfLayer][indexOfLastNeuron];
		double w = neuron.getweightsAndConnections()[0];
		double x = neuron.getWeightedSum();
		double b = neuron.getBais();

		double weightedSumOfTheOutput = w * x + b;
		chainedStored[indexOfLayer][indexOfLastNeuron] = deltacost * derivativeOfSigmoidFucntion(weightedSumOfTheOutput);

		// Update gradient
		calculatedGradient[indexOfLayer][indexOfLastNeuron][0] = chainedStored[indexOfLayer][indexOfLastNeuron] * x;

		// Momentum update for last layer weights
		vector<double>& velocities = neuron.getVelocities();
		velocities[0] = momentum * velocities[0] + learningRate * calculatedGradient[indexOfLayer][indexOfLastNeuron][0];

		// Update the weight using momentum
		neuron.getweightsAndConnections()[0] += velocities[0];

		chainedStored[indexOfLayer][indexOfLastNeuron] = deltacost * derivativeOfSigmoidFucntion(weightedSumOfTheOutput) * w * derivateOfTanhActivationFunction(x);
		indexOfLastNeuron++;
	}

	// Propagating through hidden layers
	for (int currentLayerIndex = indexOfLayer; currentLayerIndex > 0; currentLayerIndex--)
	{
		indexOfBackNeuron = 0;

		// Iterate through the neurons in the back layer
		for (int backSize = Layers[currentLayerIndex - 1].size(); indexOfBackNeuron < backSize; indexOfBackNeuron++)
		{
			indexOfFrontNeuron = 0;

			Neuron& backLayerNeuron = Layers[currentLayerIndex - 1][indexOfBackNeuron];
			vector<double>& backWeights = backLayerNeuron.getweightsAndConnections();
			vector<double>& backVelocities = backLayerNeuron.getVelocities(); // Velocity vector for the neuron
			double backWeightedSum = backLayerNeuron.getWeightedSum(); // Weighted sum (input) of this neuron
			double tempChained = 0.0;

			// Iterate through the neurons in the front layer
			for (int frontSize = Layers[currentLayerIndex].size(); indexOfFrontNeuron < frontSize; indexOfFrontNeuron++)
			{
				Neuron& frontLayerNeuron = Layers[currentLayerIndex][indexOfFrontNeuron];
				double gradient = chainedStored[currentLayerIndex][indexOfFrontNeuron] * backWeightedSum;

				// Update gradient
				calculatedGradient[currentLayerIndex - 1][indexOfBackNeuron][indexOfFrontNeuron] = gradient;

				// Momentum update for hidden layer weights
				backVelocities[indexOfFrontNeuron] = momentum * backVelocities[indexOfFrontNeuron] + learningRate * gradient;

				// Update the weight using momentum
				backWeights[indexOfFrontNeuron] += backVelocities[indexOfFrontNeuron];

				// Accumulate the chain for backpropagation
				tempChained += chainedStored[currentLayerIndex][indexOfFrontNeuron] * backWeights[indexOfFrontNeuron] * derivateOfTanhActivationFunction(backWeightedSum);
			}

			// Store the chained value for the back neuron
			chainedStored[currentLayerIndex - 1][indexOfBackNeuron] = tempChained;
		}
	}
}


void Network::backPropagationCalculateGradient(int indexOfLayer, int indexOfOutputNeurons)
{
	double momentum = 0.9;   // remove
	double learningRate = 0.5; // remove

	
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
			vector<double>& backVelocities = backLayerNeuron.getVelocities();//remove
			double tempChained = 0.0;

			//front nueron
			for (int frontSize = Layers[currentLayerIndex].size(); indexOfFrontNeuron < frontSize; indexOfFrontNeuron++)
			{
				Neuron& FrontLayerNeuron = Layers[currentLayerIndex][indexOfFrontNeuron];
				calculatedGradient[currentLayerIndex - 1][indexOfBackNeuron][indexOfFrontNeuron] = chainedStored[currentLayerIndex][indexOfFrontNeuron]/*derivative of thanh(x)*/ * backLayerNeuron.getWeightedSum()/* x of current*/;
				tempChained += (chainedStored[currentLayerIndex][indexOfFrontNeuron] * backWieght[indexOfFrontNeuron] * derivateOfTanhActivationFunction(backWeightedSum));
				
				backVelocities[indexOfFrontNeuron] = momentum * backVelocities[indexOfFrontNeuron] + learningRate;//remove

			
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
			auto& velocityM = Layers[numberofLayer][numberOfneuron].getVelocities();
			//iterate weights
			for (int numberOfWeight = 0; numberOfWeight < weightsize; numberOfWeight++)
			{
				//cout << "weight" << numberOfWeight << Layerweights[numberOfWeight]<<" + "<< calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight];
				Layerweights[numberOfWeight] -= calculatedGradient[numberofLayer][numberOfneuron][numberOfWeight];

				

			}
			cout <<endl<<"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << velocityM[numberOfneuron]<<"\n";
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

		// Push the generated pair into vectors
		tempDataInput.push_back({ static_cast<double>(x1), static_cast<double>(x2) });
		tempDataDesire.push_back(desired_output);
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


















     Network tempNet = Network(2, 4, 5, 1);
	
	vector<vector<Neuron>> Layers = tempNet.getLayers();

	//gettraining data;

	int epoch = 10000;

	auto pairedInputAndDesire = trainingGen(epoch);











	//test for right changes
	vector<vector<double>> input = pairedInputAndDesire.first;
	vector<double> desire = pairedInputAndDesire.second;

	double tempinput = 0.0;
	int count = 0;
	while (count < epoch)
	{
		
		           //step #1 set input and desired output //--------------------------------------passed!
	
		cout << "input 1st : "<< input[count].front();
		//cin >> tempinput;
		cout << endl;
		
	
		cout << "input 2nd : " << input[count].back();
		//cin >> tempinput;
		cout << endl;
		

				tempNet.setInput(input[count]);
		
		cout << "desire : "<<desire[count];
		//cin >> tempinput;
		cout << endl;
	

		tempNet.setDesiredOutput(vector<double>{desire[count]});
		



			   //step #2 feedforward.//-------------------------------------------------------passed!
			   tempNet.feedForward();

			   //step #3 calculate cost   //--------------------------------------------------passed!
			   tempNet.costFunction(BCEL);

			   //step #4 calculate cost/output

			   tempNet.derivativeOfCostFunction(BCEL);

			   //steo #5 do a gradientcalculation
			   tempNet.backPropagationCalculateGradient(5, 0);
			   //tempNet.SecondBackPropagationCalculateGradient(5, 0);//remove
			   //step #6 do a back prop
			   tempNet.backPropagationPropagate();
			   
			  // cout << "input 1: " << tempNet.getInput()[0]<<endl;
			   //cout << "input 2: " << tempNet.getInput()[1]<<endl;
			   cout << "get-output-and-cost----------------------------------------------output: " << tempNet.getOutput()[0] << " & cost: " << tempNet.getCost()<<endl;
			   tempNet.resetOutput();
			   tempNet.resetDesired();
			   tempNet.resetInput();
			   tempNet.resetCostsDeltas();
			   tempNet.resetChained();
			   tempNet.resetGradient();
			   tempNet.allocGradientAndChained();
			   count++;
	}


	



	







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