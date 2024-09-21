#include<iostream>
#include<vector>
#include<math.h>
#include<random>
using namespace std;


class Neuron {
public:
	Neuron() {};

	void makeConnections(int numberOfNextLayerNeurons);


	// temp gets
	vector<double>& getweightsAndConnections() { return w; }
	double getInput() { return x_Input; }
	double getBais() { return b_Bais; }
	double getWeightedSum() { return weightedSum; }


	// temp sets
	void set_xInput(double x) { x_Input = x; }
	void resetX() { x_Input = 0; }
	void setWeightedSum(double w) { weightedSum = w; }


private:
	//construction of activation(w*x(wx+wx+wx)+b);
	vector<double> w; //weights
	double x_Input; // X = (w+w2+w3+w4...wn)x0
	double b_Bais = 1; // bias
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
	//activations
	double tanhActivationFunction(double weightedSum) { return tanh(weightedSum); };
	double derivateOfTanhActivationFunction(double weightedSum);
	

	double sigmoidActivationFucntion(double weighted); 
	double derivativeOfSigmoidFucntion();

	double softMax();



	//costs
	void MSEcostFunction();
	void derivativeOfMSEcostFunction();
	void BinaryCrossEntropyLoss();
	void DerivativeOfBinaryCrossEntropyLoss();
	


	//temp gets
	vector<vector<Neuron>> getLayers(void) { return Layers; }
	vector<double> getOutput() { return Output; }
	vector<double> getInput() { return Input; }
	double getCost() { return cost; }
	double getDeltacost0() { return deltaCostOutput0; }

	// temp sets
	void setInput(vector<double> input) { Input = input; }
	void setOutput(vector<double> output) { Output = output; }
	void resetOutput() { Output.clear(); }
	void setDesiredOutput(vector<double> desire) { desiredOutput = desire; }


private:
	vector<vector<Neuron>> Layers;
	vector<double> Input;
	vector<double> Output;
	vector<double> desiredOutput;
	double cost;
	double deltaCostOutput0;
	double deltaCostOutput1;

};
Network::Network(int numberOfInputLayersNeuron, int numberOfHiddenLayers, int numberOfHiddenNeurons, int numberOfOutputNuerons)
{
	vector<vector<Neuron>> tempLayer(1 + numberOfHiddenLayers + 1);//supparated the inputlayer and the outputlayers

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

double Network::derivateOfTanhActivationFunction(double weightedSum)
{
	 return 1 - (tanh(weightedSum) * tanh(weightedSum)); 
}
double Network::sigmoidActivationFucntion(double weightedSum)
{
	return 1.0 / (1.0 + exp(-weightedSum));
}
void Network::feedForward()
{
	//intput // refract here so that its flexable
	Layers[0][0].set_xInput(Input[0]);
	Layers[0][1].set_xInput(Input[1]);

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

		}
	}


	//output

	weightedSum = (Layers[Layers.size() - 1][0].getweightsAndConnections()[0]/*w*/ * Layers[Layers.size() - 1][0].getInput()/*x*/ + Layers[Layers.size() - 1][0].getBais()/*b*/);
	//Layers[Layers.size() - 1][0].setWeightedSum(weightedSum);
	Output.push_back(tanhActivationFunction(weightedSum));

	weightedSum = (Layers[Layers.size() - 1][1].getweightsAndConnections()[0]/*w*/ * Layers[Layers.size() - 1][1].getInput()/*x*/ + Layers[Layers.size() - 1][1].getBais()/*b*/);
	//Layers[Layers.size() - 1][1].setWeightedSum(weightedSum);
	Output.push_back(tanhActivationFunction(weightedSum));

}

void Network::MSEcostFunction()
{
	cost = (pow(Output[0] - desiredOutput[0], 2) + pow(Output[1] - desiredOutput[1], 2)) * 0.5; //mean squered value, error
}

void Network::BinaryCrossEntropyLoss()
{
	
	cost = - (desiredOutput[0] * log(Output[0]+1e-10) + (1 - desiredOutput[0]) * log(1 - Output[0])) - (desiredOutput[1] * log(Output[1]+1e-10) + (1 - desiredOutput[1]) * log(1 - Output[1]));
	
}

void Network::DerivativeOfBinaryCrossEntropyLoss()
{

	deltaCostOutput0 = ((desiredOutput[0] / Output[0+1e-10]) - ((1 - desiredOutput[0]) / (1 - Output[0])));
	deltaCostOutput1 = ((desiredOutput[1] / Output[1]+1e-10) - ((1 - desiredOutput[1]) / (1 - Output[1])));


}

int main()
{
	vector<double> Input;
	vector<double> Output;
	vector<double> desiredOutput{vector<double>{1, 1}};





	Network tempNet = Network(2, 10, 10, 1);
	auto Layers = tempNet.getLayers();
	//tempNet.setInput(vector<double>{0.3, 0.6});
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
		cout << "deltacost0 " << tempNet.getDeltacost0();
		tempNet.resetOutput();

		
	}
}

/*
* neural network with input, output and hiddenlayer made
* connection made with all neuron in the next layer
* weight given randomly
* b = bias
* w = weight
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