#include "darknetTR.h"
#include <typeinfo>

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo){
    std::cout << "request gateway stop \n";
    gRun = false;
}

extern "C" {

void copy_image_from_bytes(image img, unsigned char *pdata){
    int w = img.w;
    int h = img.h;
    int c = img.c;

    memcpy(img.data, pdata, h*w*c);
}

image make_empty_image(int w, int h, int c){
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c){
    image out = make_empty_image(w,h,c);
    out.data = new float[h*w*c];
    return out;
}

tk::dnn::Yolo4Detection* load_network(char* net_cfg, int n_classes, int n_batch, float conf_thresh){
    std::string net;
    net = net_cfg;
    tk::dnn::Yolo4Detection *detNN = new tk::dnn::Yolo4Detection;
    detNN->init(net, n_classes, n_batch, conf_thresh);
    return detNN;
}

void do_inference(tk::dnn::Yolo4Detection *net, image img){
    std::vector<cv::Mat> batch_dnn_input;
    cv::Mat frame(img.h, img.w, CV_8UC3, (unsigned char*)img.data);
    batch_dnn_input.push_back(frame);
    net->update(batch_dnn_input, 1);    // batch_size = 1
}

void do_batch_inference(tk::dnn::Yolo4Detection *net, std::vector<image>* image_batches, int n_batch){
    std::vector<cv::Mat> batch_dnn_input;
    std::vector<image> &images = *image_batches;
    for (image img : images){
        cv::Mat frame(img.h, img.w, CV_8UC3, (unsigned char*)img.data);
        // push image into batch_dnn_input
        batch_dnn_input.push_back(frame);
    }
    // std::cout << "batch_dnn_input size: " << batch_dnn_input.size() << '\n';
    // std::cout << "batch size: " << n_batch << '\n';
    net->update(batch_dnn_input, n_batch);
}

detection* get_network_boxes(tk::dnn::Yolo4Detection *net, int batch_num, int *pnum){
    std::vector<std::vector<tk::dnn::box>> batchDetected;
    batchDetected = net->get_batch_detected();
    int nboxes = 0;
    std::vector<std::string> classesName = net->get_classesName();
    detection* dets = (detection*)calloc(batchDetected[batch_num].size(), sizeof(detection));
    
    for (int i = 0; i < batchDetected[batch_num].size(); ++i){
        dets[nboxes].cl = batchDetected[batch_num][i].cl;
        strcpy(dets[nboxes].name, classesName[dets[nboxes].cl].c_str());
        dets[nboxes].x = batchDetected[batch_num][i].x;
        dets[nboxes].y = batchDetected[batch_num][i].y;
        dets[nboxes].w = batchDetected[batch_num][i].w;
        dets[nboxes].h = batchDetected[batch_num][i].h;
        dets[nboxes].prob = batchDetected[batch_num][i].prob;
        nboxes += 1;
    }

    if (pnum) *pnum = nboxes;
    return dets;
}

result* get_batch_boxes(tk::dnn::Yolo4Detection *net){
    std::vector<std::vector<tk::dnn::box>> batchDetected;
    batchDetected = net->get_batch_detected();
    std::vector<std::string> classesName = net->get_classesName();

    result* res = new result[batchDetected.size()];

    for (int bi = 0; bi < batchDetected.size(); ++bi){
        // for each image in image batch
        detection* det = new detection[batchDetected[bi].size()];
        for (int i = 0; i < batchDetected[bi].size(); ++i){
            // for each bbox in one image
            det[i].cl = batchDetected[bi][i].cl;
            det[i].x = batchDetected[bi][i].x;
            det[i].y = batchDetected[bi][i].y;
            det[i].w = batchDetected[bi][i].w;
            det[i].h = batchDetected[bi][i].h;
            det[i].prob = batchDetected[bi][i].prob;
            strcpy(det[i].name, classesName[det[i].cl].c_str());
        }
        res[bi].dets = det;
        res[bi].nboxes = batchDetected[bi].size();
    }
    return res;
}

void freeMemory(result* res){
    delete [] res;
}

/* for passing batch images from python */
std::vector<image>* new_vector(){
    return new std::vector<image>;
}

void delete_vector(std::vector<image>* v){
    // std::cout << "destructor called in C++ for " << v << std::endl;
    std::vector<image>& vec = *v;
    for (image img : vec){
        // free the image data memory
        delete [] img.data;
    }
    delete v;
}

void vector_push_back(std::vector<image>* v, image img){
    v->push_back(img);
}

}