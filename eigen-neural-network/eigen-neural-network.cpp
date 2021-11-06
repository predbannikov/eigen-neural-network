// eigen-neural-network.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <thread>
#include <algorithm>
#include <iterator>
#include "Eigen/Dense"


using Mat = Eigen::MatrixXf;

void convertStrtoArr(std::string &str, std::vector<float> *arr, int &first)
{
	int j = 0, i, sum = 0;

	arr->at(j) = 0;
	first = str[0] - 48;
	for (i = 2; str[i] != '\0'; i++) {
		if (str[i] == ',') {
			arr->at(j) *= 0.99;
			arr->at(j) += 0.01;
			j++;
			arr->at(j) = 0;
			continue;
		}
		arr->at(j) = arr->at(j) * 10 + (str[i] - 48);
	}
	return;
}


struct NeuralNetwork {
	NeuralNetwork(int input_nodes = 5, int output_nodes = 2, int hiden_nodes = 2, float learning_rate = 0.3);
	Mat input;
	Mat wh;
	Mat hidden;
	Mat wo;
	Mat out;
	float rate;
	void forward();
	void backward(Mat& output_nodes);
	void train(const std::vector<std::pair<int, std::vector<float>*> >& vec_input);
	void query(const std::vector<std::pair<int, std::vector<float>*> >& vec_input_query);
	static void printMatrix(Mat mat, std::string name = "");
};

NeuralNetwork::NeuralNetwork(int input_nodes, int output_nodes, int hiden_nodes, float learning_rate) {
	input = Mat(input_nodes, 1);
	out = Mat(output_nodes, 1);
	hidden = Mat(hiden_nodes, 1);
	wh = Mat::Random(hiden_nodes, input_nodes) * 0.5;
	wo = Mat::Random(output_nodes, hiden_nodes) * 0.5;
	rate = learning_rate;
}

inline void NeuralNetwork::forward() {
	auto sigmoid = [](const float z) -> float { return 1.0 / (1.0 + exp(-z)); };
	hidden = wh * input;
	hidden = hidden.unaryExpr(sigmoid);
	//printMatrix(hidden, "whiddenh");
	out = wo * hidden;
	out = out.unaryExpr(sigmoid);
}

inline void NeuralNetwork::query(const std::vector<std::pair<int, std::vector<float>*> >& vec_input_query) {
	for (auto it = vec_input_query.begin(); it != vec_input_query.end(); it++) {
		const std::vector<float>& v = *(*it).second;
		int target = (*it).first;

		for (int i = 0; i < v.size(); i++)
			input(i, 0) = v[i];

		forward();
		std::cout << "test:\t" << target << std::endl;
		printMatrix(out);
	}

}

void NeuralNetwork::train(const std::vector<std::pair<int, std::vector<float>*> >& vec_input)
{
	for (auto it = vec_input.begin(); it != vec_input.end(); it++) {
		const std::vector<float> &v = *(*it).second;
		int target = (*it).first;

		for (int i = 0; i < v.size(); i++) 
			input(i, 0) = (v[i] / 255.) * 0.99 + 0.01;

		Mat target_output = Mat::Zero(10, 1) + Mat::Constant(10, 1, 0.01);
		target_output(target, 0) = 0.99;

		forward();
		backward(target_output);
	}
}

void NeuralNetwork::backward(Mat& target_list)
{
	Mat out_errors = target_list - out;
	Mat hidden_errors = wo.transpose() * out_errors;

	wo += rate * Mat(out_errors.array() * out.array() * (Mat::Constant(out.rows(), 1, 1.0) - out).array()) * hidden.transpose();
	wh += rate * Mat(hidden_errors.array() * hidden.array() * (Mat::Constant(hidden.rows(), 1, 1.0) - hidden).array()) * input.transpose();
}

void NeuralNetwork::printMatrix(Mat mat, std::string name)
{
	int row = mat.rows();
	int col = mat.cols();
	std::string str;
	std::cout << "PRINT " << name << std::endl;
	for (size_t i = 0; i < row; i++) {
		for (size_t j = 0; j < col; j++) {
			str += std::to_string(mat(i, j)) + "\t";
		}
		std::cout << str << std::endl;;
		str.clear();
	}
}

void loadSet(std::string name, std::vector<std::pair<int, std::vector<float>*>>& vec) {

	std::fstream file;
	file.open(name, std::ios::in);
	if (!file.is_open()) {
		std::cout << "file not open" << std::endl;
		return;
	}
	std::string line;
	vec.reserve(60000);
	std::stringstream buffer;
	buffer << file.rdbuf();
	file.close();
	int first_sign;
	//std::string::size_type prev_pos, pos;
	//long mark, dmark;
	long markStart = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	while (std::getline(buffer, line))
	{
		// METHOD 1
		//mark = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		//std::vector<float> *v = new std::vector<float>(784);
		//dmark = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - mark;
		////std::cout << "create vector " << dmark << std::endl;
		//mark = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		////v->reserve(784);
		////std::getline(s, first_sign, ',');
		//prev_pos = pos = 0;
		//pos = line.find(',', pos);
		//first_sign = std::atoi(line.substr(prev_pos, pos - prev_pos).c_str());
		//prev_pos = ++pos;
		//dmark = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - mark;
		////std::cout << "commont spent " << dmark << std::endl;
		//mark = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		//for(int i = 0; i < 784; i++) { // (std::getline(s, field, ',')) 
		//	pos = line.find(',', pos);
		//	v->at(i) = 0.99 * std::atoi(line.substr(prev_pos, pos-prev_pos).c_str()) + 0.01;
		//	prev_pos =  ++pos;
		//}
		//dmark = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - mark;
		//std::cout << "end cycle " << dmark << std::endl;

		// METHOD 2
		//std::istringstream s(line);
		//std::string field;
		//std::vector<float> *v = new std::vector<float>(784);
		//v->reserve(784);
		//std::string s_tmp;
		//getline(s, s_tmp, ',');
		//first_sign = std::atoi(s_tmp.c_str());
		//while (getline(s, field, ',')) 
		//std::stringstream stream(field);
		//	v->push_back(0.99 * std::atoi(field.c_str()) + 0.01);
		//map.push_back({first_sign, v});

		// METHOD 3
		std::vector<float>* v = new std::vector<float>(784);
		convertStrtoArr(line, v, first_sign);
		vec.push_back({ first_sign, v });

	}
	// 78 sec with find method
	// 62 sec second method getline
	// 25 sec third method convertStrToArr
	std::cout << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count() - markStart << "sec. - time loading" << std::endl;
}

int main()
{
	std::string name1 = "H:\\projects\\neuralnetwork\\mnist_train_100.csv";
	std::string name2 = "H:\\projects\\neuralnetwork\\mnist_test_10.csv";
	std::vector<std::pair<int, std::vector<float>*>> vec_train;
	std::vector<std::pair<int, std::vector<float>*>> vec_query;
	loadSet(name1, vec_train);
	loadSet(name2, vec_query);
    NeuralNetwork n(784, 10, 100, 0.3);
	n.train(vec_train);
	n.query(vec_query);

}

