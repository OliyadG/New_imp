#include<iostream>
#include<vector>
#include<math.h>
#include<random>
using namespace std;





class Neuron {
public:
	Neuron() {};

	void makeConnections(int numberOfNextLayerNeurons);
	double static activationFunction(double weightedSum) { return tanh(weightedSum); };


	// temp gets
	vector<double>& getweightsAndConnections() { return w; }
	double getInput() { return x_Input; }
	double getBais() { return b_Bais; }
	double getY_activated() { return y_Activated; }


	// temp sets
	void set_xInput(double x) { x_Input = x; }
	void resetX() { x_Input = 0; }


private:
	//construction of activation(w*x(wx+wx+wx)+b);
	vector<double> w; //weights
	double x_Input; // X = (w+w2+w3+w4...wn)x0
	double b_Bais = 1; // bias
	double y_Activated; // output

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
	double getWeightedSum(int layerIndex, int NueronIndex);
	void feedForward();


	//temp gets
	vector<vector<Neuron>> getLayers(void) { return Layers; }
	vector<double> getOutput() { return Output; }
	vector<double> getInput() { return Input; }

	// temp sets
	void setInput(vector<double> input) { Input = input; }
	void setOutput(vector<double> output) { Output = output; }
	void resetOutput() { Output.clear(); }


private:
	vector<vector<Neuron>> Layers;
	vector<double> Input;
	vector<double> Output;

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



void Network::feedForward()
{
	//intput
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
			double Activated = Neuron::activationFunction(weightedSum);

			//cout <<"\n" << Layers[singleLayer + 1][singleNeuron].getInput() << "******************************** start" << endl;
			Layers[singleLayer + 1][singleNeuron].set_xInput(Activated);
			//cout<<"\n" << Layers[singleLayer + 1][singleNeuron].getInput() << "****************************** end" << endl;

			//cout <<"Weightedsum: ="<<weightedSum<<endl<< "Activated: = " << Activated << endl<<endl;

		}
	}
	//output
	weightedSum = (Layers[Layers.size() - 1][0].getweightsAndConnections()[0]/*w*/ * Layers[Layers.size() - 1][0].getInput()/*x*/ + Layers[Layers.size() - 1][0].getBais()/*b*/);
	Output.push_back(Neuron::activationFunction(weightedSum));

	weightedSum = (Layers[Layers.size() - 1][1].getweightsAndConnections()[0]/*w*/ * Layers[Layers.size() - 1][1].getInput()/*x*/ + Layers[Layers.size() - 1][1].getBais()/*b*/);
	Output.push_back(Neuron::activationFunction(weightedSum));

}

int main()
{
	Network tempNet = Network(2, 10, 10, 2);
	auto Layers = tempNet.getLayers();
	tempNet.setInput(vector<double>{0.3, 0.6});

	while (true) {

		double x, y;
		cin >> x >> y;
		tempNet.setInput(vector<double>{x, y});
		tempNet.feedForward();

		cout << "outputNuron1: " << tempNet.getOutput()[0];
		cout << "\noutputNuron2: " << tempNet.getOutput()[1] << endl;
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



*/