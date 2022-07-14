#ifndef DEMO_H
#define DEMO_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <mutex>
#include <malloc.h>
#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo4Detection.h"
#include "utils.h"

extern "C" {

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct {
    float x, y, w, h;
} BOX;

typedef struct {
    // store each image's detection
    int cl;
    float x, y, w, h;
    float prob;
    char name[20];
} detection;

typedef struct {
    // store each batch's detection
    detection* dets;
    int nboxes;
} result;

tk::dnn::Yolo4Detection* load_network(char* net_cfg, int n_classes, int n_batch, float conf_thresh);
}

#endif /* DETECTIONNN_H*/
