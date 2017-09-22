#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <opencv2/opencv.hpp>
#define USE_OPENCV
#define HAVE_CUDA
#define USE_CUDNN
#include <caffe/net.hpp>
#include <caffe/layers/memory_data_layer.hpp>

using namespace std;

void addtofiledfc2017(const string& name, const string& sentinelname, const string& root){
	ofstream file_((name +".txt").c_str());
	ofstream * file = &file_;
	
	cout << "######################## " << name << " ###########################" << endl;
	
	cv::Mat vt = cv::imread(root+name+"/lcz/"+name+"_lcz_GT.tif",CV_LOAD_IMAGE_GRAYSCALE);
	cout << "load vt " << vt.cols << ' ' << vt.rows << endl;
	
	vector<cv::Mat> sentinel;
	if(true){
		vector<string> bande;
		bande.push_back("B02");
		bande.push_back("B03");
		bande.push_back("B04");
		bande.push_back("B05");
		bande.push_back("B06");
		bande.push_back("B07");
		bande.push_back("B08");
		bande.push_back("B11");
		bande.push_back("B12");
		
		sentinel = vector<cv::Mat>(bande.size());
		for(int i=0;i<(int)bande.size();i++){
			sentinel[i] = cv::imread(root+name+"/sentinel_2/"+sentinelname+bande[i]+".tif",CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_GRAYSCALE);
			cout << root+name+"/sentinel_2/"+sentinelname+bande[i]+".tif" << ' ' << sentinel[i].cols << ' ' << sentinel[i].rows << ' ' << sentinel[i].type() << endl;
		}
	}
	
	vector<vector<cv::Mat> > landsat;
	if(true){
		vector<string> bandelandsat;
		bandelandsat.push_back("B1");
		bandelandsat.push_back("B2");
		bandelandsat.push_back("B3");
		bandelandsat.push_back("B4");
		bandelandsat.push_back("B5");
		bandelandsat.push_back("B6");
		bandelandsat.push_back("B7");
		bandelandsat.push_back("B10");
		bandelandsat.push_back("B11");
		
		vector<string> landsatradix;
		if(true){
			const int sysr = system(("ls "+root+name+"/landsat_8 > tmp.txt").c_str());
			ifstream tmpfile("tmp.txt");
			string line;
			while(getline(tmpfile,line))
				if(line!="")
					landsatradix.push_back(line);
			cout << "found ";
			for(int i=0;i<(int)landsatradix.size();i++)
				cout << landsatradix[i] << ' ';
			cout << endl;
		}
		
		landsat = vector<vector<cv::Mat> >(landsatradix.size(),vector<cv::Mat>(bandelandsat.size()));
		for(int i=0;i<(int)landsatradix.size();i++)
			for(int j=0;j<(int)bandelandsat.size();j++){
				landsat[i][j] = cv::imread(root+name+"/landsat_8/"+landsatradix[i]+"/"+landsatradix[i]+"_"+bandelandsat[j]+".tif",CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_GRAYSCALE);
				cout << "landsat " << landsatradix[i] << ' ' << bandelandsat[j] << ' ' << landsat[i][j].cols << ' ' << landsat[i][j].rows << ' ' << landsat[i][j].type() << endl;
			}
	}
	
	vector<cv::Mat> imagefromvgg;
	const string pathproto="../vgg7bis.prototxt";
	vector<string> layercnn;
	layercnn.push_back("conv3_1");
	layercnn.push_back("conv3_3");
	vector<int> vggsize;
	vggsize.push_back(256);
	vggsize.push_back(256);
	vector<int> vggpool;
	vggpool.push_back(16);
	vggpool.push_back(64);
	for(int layerI=0;layerI<(int)layercnn.size();layerI++){
		cv::Mat imageforvggraw = cv::imread(root+name+"/osm_raster/rvb.png");
		int cols1024,rows1024;
		if(imageforvggraw.cols%1024==0)
			cols1024=imageforvggraw.cols;
		else
			cols1024=(imageforvggraw.cols/1024)*1024+1024;
		if(imageforvggraw.rows%1024==0)
			rows1024=imageforvggraw.rows;
		else
			rows1024=(imageforvggraw.rows/1024)*1024+1024;		
		
		cv::Mat imageforvgg1024;
		cv::resize(imageforvggraw,imageforvgg1024,cv::Size(cols1024,rows1024));
		imageforvggraw = cv::Mat(0,0,0);
		
		const int DC = imageforvgg1024.cols/1024;
		const int DR = imageforvgg1024.rows/1024;
		vector<vector<cv::Mat> > grid(DC,vector<cv::Mat>(DR));
		for(int dc=0;dc<DC;dc++)
			for(int dr=0;dr<DR;dr++)
				grid[dc][dr] = (imageforvgg1024(cv::Rect(dc*1024,dr*1024,1024,1024))).clone();
		imageforvgg1024 = cv::Mat(0,0,0);
		
		vector<vector<vector<cv::Mat> > > gridaftercnn(grid.size(),vector<vector<cv::Mat> >(grid[0].size(),vector<cv::Mat>(vggsize[layerI])));
		for(int dc=0;dc<DC;dc++)
			for(int dr=0;dr<DR;dr++)
				for(int ch=0;ch<vggsize[layerI];ch++)
					gridaftercnn[dc][dr][ch] = cv::Mat(1024/vggpool[layerI],1024/vggpool[layerI],CV_32F,cv::Scalar::all(0));
					
		caffe::shared_ptr<caffe::Net<float> > cnn;
		cnn = caffe::shared_ptr<caffe::Net<float> >(new caffe::Net<float>(pathproto,caffe::TEST));
		cnn->CopyTrainedLayersFrom("/media/achanhon/bigdata/data/imagenet/vgg16.caffemodel");
				
		for(int dc=0;dc<DC;dc++)
			for(int dr=0;dr<DR;dr++){
				vector<int> dumbLabel(1,0);	
				vector<cv::Mat> dumbbatch(1,grid[dc][dr]);
				boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(cnn->layers()[0])->AddMatVector(dumbbatch,dumbLabel);
				cnn->ForwardFrom(0);
				
				caffe::shared_ptr<caffe::Blob<float> > feature_blob = cnn->blob_by_name(layercnn[layerI]);
				float* feature_blob_data = feature_blob->mutable_cpu_data();
				for(int ch=0;ch<vggsize[layerI];ch++)
					for(int row=0;row<1024/vggpool[layerI];row++)
						for(int col=0;col<1024/vggpool[layerI];col++)
							gridaftercnn[dc][dr][ch].at<float>(row,col) = feature_blob_data[feature_blob->offset(0,ch,row,col)];
			}
		grid = vector<vector<cv::Mat> >();
		
		vector<cv::Mat> imagefromvgg1024(vggsize[layerI]);
		for(int i=0;i<vggsize[layerI];i++)
			imagefromvgg1024[i] = cv::Mat(cv::Size(cols1024/vggpool[layerI],rows1024/vggpool[layerI]),CV_32F,cv::Scalar::all(0));
		
		for(int ch=0;ch<vggsize[layerI];ch++)
			for(int dc=0;dc<DC;dc++)
				for(int dr=0;dr<DR;dr++)
					for(int row=0;row<1024/vggpool[layerI];row++)
						for(int col=0;col<1024/vggpool[layerI];col++)
							imagefromvgg1024[ch].at<float>(dr*1024/vggpool[layerI]+row,dc*1024/vggpool[layerI]+col) = gridaftercnn[dc][dr][ch].at<float>(row,col);
		gridaftercnn = vector<vector<vector<cv::Mat> > >();
							
		for(int ch=0;ch<vggsize[layerI];ch++){
			cv::Mat tmp;
			cv::resize(imagefromvgg1024[ch],tmp,vt.size());
			imagefromvgg.push_back(tmp);
		}
	}
	
	struct local{
		static float means(const vector<float>& in){
			float out=0;
			for(int i=0;i<(int)in.size();i++)
				out+=in[i];
			out/=in.size()+in.empty();
			return out;
		}
		static float var(const vector<float>& in){
			const float m = means(in);
			float out=0;
			for(int i=0;i<(int)in.size();i++)
				out+=(in[i]-m)*(in[i]-m);
			out/=in.size()+in.empty();
			out = sqrt(out);
			return out;
		}
		static void addtosvm(ofstream * file, const int y, const vector<float>& x){
			(*file) << y;
			for(int i=0;i<(int)x.size();i++)
				(*file) << ' ' << i+1 << ':' << x[i];
			(*file) << '\n';
		}
	};
	
	vector<cv::Mat> landsatmean(landsat[0].size());
	vector<cv::Mat> landsatvar(landsat[0].size());
	for(int i=0;i<(int)landsat[0].size();i++){
		landsatmean[i] = cv::Mat(vt.size(),CV_32F);
		landsatvar[i] = cv::Mat(vt.size(),CV_32F);
		for(int r=0;r<vt.rows;r++)
			for(int c=0;c<vt.cols;c++){
				vector<float> all(landsat.size());
				for(int j=0;j<(int)all.size();j++)
					all[j] = landsat[j][i].at<float>(r,c);
					
				landsatmean[i].at<float>(r,c) = local::means(all);
				landsatvar[i].at<float>(r,c) = local::var(all);
			}
	}
	
	for(int r=0;r<vt.rows;r++)
		for(int c=0;c<vt.cols;c++)
			if(vt.data[r*vt.cols+c]!=0){
				vector<float> all;
				for(int i=0;i<(int)sentinel.size();i++)
					all.push_back(sentinel[i].at<float>(r,c));
				for(int i=0;i<(int)landsatmean.size();i++){
					all.push_back(landsatmean[i].at<float>(r,c));
					all.push_back(landsatvar[i].at<float>(r,c));
				}
				for(int i=0;i<(int)imagefromvgg.size();i++)
					all.push_back(imagefromvgg[i].at<float>(r,c));
				local::addtosvm(file,vt.data[r*vt.cols+c],all);
			}
}

int main(int argc,char **argv){
	google::InitGoogleLogging("");
	caffe::Caffe::SetDevice(0);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	
	string root = "/data/DFC2017/train/";
	addtofiledfc2017("paris","Paris_S2_",root);			
	addtofiledfc2017("berlin","Berlin_S2_",root);		
	addtofiledfc2017("rome","",root);					
	addtofiledfc2017("hong_kong","HongKong_S2_",root);	
	addtofiledfc2017("sao_paulo","SaoPaulo_S2_",root);	
	return 0;
}
