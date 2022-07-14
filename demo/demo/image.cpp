#include <iostream>
#include <signal.h>
#include <stdlib.h>     
#include <unistd.h>
#include <mutex>

#include "Yolo3Detection.h"
#include <chrono>

using namespace std::chrono; 


int main(int argc, char *argv[]){

    // Network name
    std::string net = "yolo4_fp16.rt";
    if(argc > 1)
        net = argv[1];
    
    // Inference batch size 
    int n_batch = 4;
    if(argc > 2)
        n_batch = atoi(argv[2]);
        
    // Number of classes
    int n_classes = 3;
    if(argc > 3)
        n_classes = atoi(argv[3]);
  
    tk::dnn::Yolo3Detection yolo;  

    tk::dnn::DetectionNN *detNN; 
    detNN = &yolo;
    detNN->init(net, n_classes, n_batch);	
    
    cv::Mat img;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    auto start = high_resolution_clock::now(); 
    // cv::namedWindow("detection", cv::WINDOW_NORMAL);  

    // Read all images from a directory          
    cv::String path("../images/*.*");
    std::vector<cv::String> fn;
    cv::glob(path,fn,true); // recurse
    int total = 0;
    int num_img = 0; 
    for (size_t k=0; k < fn.size(); ++k){
        img = cv::imread(fn[k], cv::IMREAD_COLOR);
        if (!img.data){
            std::cout << "Problem loading image!";
            break;
        } 
        batch_frame.push_back(img);

        // this will be resized to the net format
        batch_dnn_input.push_back(img.clone());
        ++num_img;
        
        
        // do inference
        if (num_img == n_batch){
            detNN->update(batch_dnn_input, n_batch);
            detNN->draw(batch_frame); 
            for(int bi=0; bi< n_batch; ++bi){
            //    cv::imshow("detection", batch_frame[bi]);
            //    cv::waitKey(10);
                ++total;
            }           
            num_img = 0;
            batch_dnn_input.clear();
            batch_frame.clear();
        }            
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);  
    double mean = 0; 
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout << "Total files: " << total << " files\n";
    std::cout << "Total inference time: " << duration.count() << "s\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END; 
    
}