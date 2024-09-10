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
	void static activationFunction() {};
	

private:
	//construction of activation(w*x(wx+wx+wx)+b);

	vector<double> w;
	double x; // X = (w+w2+w3+w4...wn)x0
	double b;

	double y;


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
	vector<vector<Neuron>> getLayers(void) { return Layers; }

private:

	vector<vector<Neuron>> Layers;

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


int main()
{
	Network tempNet = Network(2, 10, 500, 2);

	auto Layers = tempNet.getLayers();

	size_t size_in_bytes = Layers.size() * sizeof(int);
	double size_in_megabytes = static_cast<double>(size_in_bytes) / (1024 * 1024);


	std::cout << "Size of the vector in bytes: " << size_in_bytes << std::endl;
	std::cout << "Size of the vector in megabytes: " << size_in_megabytes << " MB" << std::endl;


}