#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;

/*
* Notes:
* This file is run using OpenCV 3.8.1 library and MNIST datasets
* The image processing model is made using K-Nearest Neighbor
*/

/*
* The function convertEndian serves to swap the endianness of 32 bit unsigned integers
*/

uint32_t convertEndian(uint32_t value) {
    return ((value >> 24) & 0xFF) | ((value >> 8) & 0xFF00) | ((value << 8) & 0xFF0000) | ((value << 24) & 0xFF000000);
}

/*
* Implementation notes: readLabel, readImage
* The readLabel function serves to read the MNIST idx3 label binary files and put the label information in the vectors
* 
* The readImage function serves to read the MNIST idx3 image binary files and attain the pixel datas of every images in the file.
*/

vector<uint8_t> readLabel(string filename)
{
    ifstream file(filename, ios::binary);
    if (!file)
    {
        cout << "Error: File does not exist!" << endl;
        return vector<uint8_t>();
    }
    int32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    int32_t num_labels;
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = convertEndian(num_labels);
    vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    file.close();
    return labels;
}

vector< vector <uint8_t> > readImage(string filename, int &row, int &col)
{
    ifstream file(filename, ios::binary);
    if (!file)
    {
        cout << "Error: File does not exist!" << endl;
        return vector<vector<uint8_t>>();
    }
    int32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    int32_t num_image;
    file.read(reinterpret_cast<char*>(&num_image), sizeof(num_image));
    num_image = convertEndian(num_image);
    int32_t num_row;
    file.read(reinterpret_cast<char*>(&num_row), sizeof(num_row));
    num_row = convertEndian(num_row);
    int32_t num_col;
    file.read(reinterpret_cast<char*>(&num_col), sizeof(num_col));
    num_col = convertEndian(num_col);

    vector<uint8_t> image_pixel(num_row * num_col); // This vector is used to store every single pixel of a single image
    vector<vector <uint8_t>> images; // This vector a vector for the image_pixel vector

    for (int i = 0; i < num_image; i++)
    {
        file.read(reinterpret_cast<char*>(image_pixel.data()), num_row * num_col);
        images.push_back(image_pixel);
    }
    row = num_row;
    col = num_col;
    return images;
}

/*
* Implementation notes: preprocessImage
* Once the image binary files has been decoded, the pixels information of every image is converted to CV_32F format to make model training more efficient and easier.
* Returns matrix of an preprocessed image, where it would be stored in a vector of image matrices.
* Example: 255 255 255 0 0 0 -> 1 1 1 0 0 0
*/

Mat preprocessImage(const vector<uint8_t> &image, int row, int col)
{
    Mat image_result(1, row * col, CV_32F);
    for (int i = 0; i < row * col; i++)
    {
        image_result.at<float>(0, i) = image[i] / float(255);
    }
    return image_result;
}

/*
* Implementation notes: modelTraining, predict
* Once the training dataset finished preprocessing, K-Nearest Neighbor model from OpenCV library begins training.
* Before training, every image is compressed into a row vector where it would be appended to a matrix as each column, i.e every row in training_dataset is a representation of an image
* The label vector is also converted into a row matrix, where both training_dataset and labels_mat is used to train the KNN model
* Example: training_dataset
* After training has finished, the model is ready to be used to predict the test datasets
* Each input in predict function is in form of an image
*/

Ptr<KNearest> modelTraining(vector<Mat> &images, vector<uint8_t> &labels)
{
    Ptr<KNearest> knn = KNearest::create();
    Mat training_dataset;
    for (int i = 0; i < labels.size(); i++)
    {
        Mat img_pixel_row = images[i].reshape(1, 1);
        training_dataset.push_back(img_pixel_row);
    }
    Mat labels_mat(labels);
    training_dataset.convertTo(training_dataset, CV_32F);
    labels_mat.convertTo(labels_mat, CV_32S);
    knn->train(training_dataset, ROW_SAMPLE, labels_mat);
    return knn;
}

int predict(Ptr<KNearest> &model, Mat image)
{
    image.convertTo(image, CV_32F);
    image.reshape(1, 1);
    Mat result, response, dist;
    float num = model->findNearest(image, 1, result, response, dist);
    int predict = static_cast<int>(num);
    return predict;
}

int main()
{
    int row, col; // Variables to indicate the size of rows and columns
    
    // Training dataset preparation
    string training_label_file = "train-labels.idx1-ubyte";
    string training_image_file = "train-images.idx3-ubyte";
    vector<uint8_t> train_label_list = readLabel(training_label_file);
    vector< vector<uint8_t> > train_image_matrix = readImage(training_image_file, row, col);
    if (train_label_list.empty() || train_image_matrix.empty()) // Error handling if training datasets are unavailable
    {
        cout << "Insufficient data!" << endl;
        return 1;
    }

    // Model training session
    vector<Mat> image_training;
    for (int i = 0; i < train_image_matrix.size(); i++)
    {
        image_training.push_back(preprocessImage(train_image_matrix[i], row, col));
    }
    Ptr<KNearest> model = modelTraining(image_training, train_label_list);

    // Test dataset preparation
    string test_label = "t10k-labels.idx1-ubyte";
    string test_image = "t10k-images.idx3-ubyte";
    vector<uint8_t> test_label_list = readLabel(test_label);
    vector< vector<uint8_t> > test_image_matrix = readImage(test_image, row, col);
    if (test_label_list.empty() || test_image_matrix.empty()) // Error handling if test datasets are unavailable
    {
        cout << "Insufficient test data!" << endl;
        return 1;
    }
    vector<Mat> image_test;
    for (int i = 0; i < test_image_matrix.size(); i++)
    {
        image_test.push_back(preprocessImage(test_image_matrix[i], row, col));
    }

    // Model testing session
    int accurate = 0;
    int prediction;
    for (int i = 0; i < image_test.size(); i++)
    {
        Mat imageToShow = image_test[i].reshape(1, row);
        resize(imageToShow, imageToShow, Size(400,400), INTER_LINEAR);
        imshow("Image", imageToShow);
        prediction = predict(model, image_test[i]);
        cout << "Test Number " << i + 1 << endl;
        cout << "Predicted Number: " << prediction << endl;
        cout << "Actual Number: " << static_cast<int>(test_label_list[i]) << endl;
        cout << endl;
        if (prediction == test_label_list[i])
        {
            accurate++;
        }
        waitKey(1);
    }
    float accuracy = accurate/(float)100; // Divide the number of tested image by 100
    cout << "Accuracy: " << accuracy << "%" << endl;
    return 0;
}
