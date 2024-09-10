#include<iostream>
#include<vector>
using namespace std;


class Neuron {
public:
	Neuron() {};

	void makeConnections(int numberOfNextLayerNeurons);
	void getweightsAndConnections() {
		for (auto ss : w)
		{
			cout << ss << endl;
		}
	}
	void static activationFunction() {};

private:
	//construction of activation(w*x(wx+wx+wx)+b);
	vector<double> w;


	double x_Input; // X = (w+w2+w3+w4...wn)x0
	double b_Bais = 1; // bias
	double y_Activated; // output


	int myIndex;
};
void Neuron::makeConnections(int numberOfNextLayerNeurons)
{
	w = vector<double>(numberOfNextLayerNeurons);

	for (auto& tempCon :w)
	{
		tempCon = rand() / double(RAND_MAX);
	}
}


class Network {

public:
	Network(int numberOfInputLayerNeuron, int numberOfHiddenLayers, int numberOfNeurons, int numberOfOutputNeurons);
	void feedForward();



	//temp sets and gets
	vector<vector<Neuron>> getLayers(void) { return Layers; }
	void setInput(vector<double> input) { Input = input; }


private:
	vector<vector<Neuron>> Layers;
	static vector<double> Input;

};
Network::Network(int numberOfInputLayersNeuron, int numberOfHiddenLayers, int numberOfHiddenNeurons, int numberOfOutputNuerons)
{
	vector<vector<Neuron>> tempLayer(1+  numberOfHiddenLayers  +1);//supparated the inputlayer and the outputlayers

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
		tempLayer[numberOfHiddenLayers+1].push_back(tempNeuron);//+2 because we have added extra 2 conteners for the input and output!, the input took the [0], the out put took[+1]
	}
	Layers = tempLayer;
}

void Network::feedForward()
{


	

	for (int singleLayer = 0;singleLayer < Layers.size(); singleLayer++)
	{
		
		for (int singleNeuron = 0; singleNeuron < Layers[singleLayer].size(); singleNeuron++)
		{
			//y = w*x+b


		}

	}
}

int main()
{
	Network tempNet = Network(2, 10, 10, 2);
	auto Layers = tempNet.getLayers();

	tempNet.setInput(vector<double>{1.0, 0.0});
	


	//size printer
	tempNet.feedForward();

	//check Made neurons fo far with weight
	int count = 0;
	for (auto& aLayer : Layers)
	{
		cout << "***layer: " << count << "***" << endl;
		for (auto& Nuro : aLayer)
		{
			Nuro.getweightsAndConnections();
			cout << "****DONe*****\n";
		}
		count++;
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
* missing, feedforward, the delta



*/