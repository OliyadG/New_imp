#include<iostream>
#include<vector>

using namespace std;

struct Connections {

	double weight;
};


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

private:
	//construction of activation(w*x(wx+wx+wx)+b);

	vector<double> w;
	double x; // X = (w+w2+w3+w4...wn)x0
	double b;

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

	Network(int numberOfInputLayers, int numberOfHiddenLayers, int numberOfNeurons, int numberOfOutputNeurons);
	vector<vector<Neuron>> getLayers(void) { return Layers; }

private:

	vector<vector<Neuron>> Layers;

};

Network::Network(int numberOfInputLayers, int numberOfHiddenLayers, int numberOfNeurons, int numberOfOutputNuerons)
{
	vector<vector<Neuron>> tempLayer(numberOfHiddenLayers);

	for (int countInputLayers = 0; countInputLayers < numberOfInputLayers; countInputLayers++)
	{
		Neuron tempNeuron = Neuron();
		tempNeuron
		tempLayer[0]fdjj
	}


	for (int countLayers = 1; countLayers < numberOfHiddenLayers; countLayers++)
	{
		for (int countNeurons = 0; countNeurons < numberOfNeurons; countNeurons++)
		{
			Neuron tempNeuron = Neuron();
			tempNeuron.makeConnections(numberOfNeurons);
			tempLayer[countLayers].push_back(tempNeuron);

		}

		
	}
	
	//outputLayer
	for (int outputNeurons = 0; outputNeurons <= numberOfOutputNuerons; outputNeurons++) {
		Neuron tempNeuron = Neuron();
		tempNeuron.makeConnections(numberOfNeurons);
		tempLayer[numberOfHiddenLayers].push_back(tempNeuron);
	}
}


int main()
{
	Network tempNet = Network(2, 3, 5, 2);

	auto Layers = tempNet.getLayers();

	//check Made neurons fo far with weight
	int count = 0;
	for (auto &aLayer : Layers)
	{
		cout << "***layer: " << count <<"***" << endl;
		for (auto& Nuro : aLayer)
		{
			Nuro.getweightsAndConnections();
			cout << "****DONe*****\n";
		}
	}


}