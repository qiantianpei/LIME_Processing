#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <lime.h>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <string>

using namespace std;
using namespace Json;

int main()
{
    ifstream ifs("/home/tianpei/workspace/data_local/us/us_test_1.1.json");
    Reader reader;
    Value obj;
    reader.parse(ifs, obj);
    string data_path = "/home/tianpei/workspace/data_local/us/", img_path;

    cv::Mat img_in, img_out;

    for (int i = 0; i < obj.size(); ++i) {
        img_path = obj[i]["file"].asString();
        img_in = cv::imread(data_path + img_path);

        if (img_in.empty()) {
            std::cout<<"Error Input!"<<std::endl;
            return -1;
        }

        unique_ptr<feature::lime> l(new feature::lime(img_in));
        img_out = l->lime_enhance(img_in);
        // cvNamedWindow("raw_picture",CV_WINDOW_NORMAL);
        // cvNamedWindow("img_lime",CV_WINDOW_NORMAL);
        // imshow("raw_picture",img_in);
        // imshow("img_lime",img_out);
        int last_dot_pos = img_path.rfind('.');
        cout << data_path + img_path.substr(0, last_dot_pos) + ".lime" + img_path.substr(last_dot_pos) << endl;
        cv::imwrite(data_path + img_path.substr(0, last_dot_pos) + ".lime" + img_path.substr(last_dot_pos), img_out);
    }    

    return 0;
}


