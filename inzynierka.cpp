#include <opencv2/opencv.hpp>
#include <iostream>
#include <X11/Xlib.h>
#include <stdio.h>
#include <vector>
#include <thread>
#include <unistd.h>
#include<fstream>
#include <mutex>
using namespace std;
using namespace cv;

//zmienne globalne
Mutex blokada;
Point oko_lewe=Point(0,0);
Point oko_prawe=Point(0,0); 
Point srodek;
Point  przefiltrowane_x;
Point przefiltrowane_l;
int t=0;
vector <int> ruch_poziomy(2);
vector <int> ruch_pionowy(2);
//ruch_poziomy[0]=0;
//ruch_pionowy[0]=0;

//parametry do znalezeinia koloru
int hmax=255;
int hmin=1;
int smax=255;
int smin=1;
int vmax=48;
int vmin=1;
//macierze
Mat na_zywo, dylatacja, erozja, Blur;
Mat mask;
Mat hsv;
Mat szare_zdjecie;
Mat kontury;
vector <Mat> uciete;
Mat konturki;
//kaskady
CascadeClassifier cascade;
CascadeClassifier kaskada_oczu;
bool mrugniecie();
Point wyliczenie_sredniej (char a);

//Pierwszy wątek znajdujący źrenice i zapisujący jej położenie
void f1()
{    
    kaskada_oczu.load("haarcascade_eye.xml");
    cascade.load("haarcascade_frontalface_default.xml");
    //zaznaczenie twarzy i oczu
    vector <Rect> twarze;
    vector <int> pomocnicze;
    vector <Rect> oczy;
    Scalar kolor1=Scalar(0,255,0);
    Scalar kolor=Scalar(255,0,0);
    
    //zaznaczenie źrenic
    vector <Vec3f> linie;

    //Przechwycenie obrazu
    VideoCapture przechwycenie=VideoCapture(0);

    //przechwycenie  obrazu na żywo
    if(!przechwycenie.open(0))
        CV_Assert("Kamera nie działa");
    //przechwycenie.set(CAP_PROP_FRAME_WIDTH, 1280);
    //przechwycenie.set(CAP_PROP_FRAME_HEIGHT, 720);

    // ciachniecie
    Rect wycinek;
    Rect kwadrat;
    /*
    //Sprawcdzenie jakie dobrać parametry
    namedWindow("Trackedbar", (500,300));
    createTrackbar("hmin","Trackedbar", &hmin,255);
    createTrackbar("hmax","Trackedbar", &hmax,255);
    createTrackbar("smin","Trackedbar", &smin,255);
    createTrackbar("smax","Trackedbar", &smax,255);
    createTrackbar("vmin","Trackedbar", &vmin,255);
    createTrackbar("vmax","Trackedbar", &vmax,255);
    // na_zywo=imread("oczy3.jpg");
    while(true)
    {
        przechwycenie>>na_zywo; 
        cvtColor(na_zywo, hsv, COLOR_BGR2HSV);
        Scalar lower(hmin, smin, vmin);
        Scalar upper (hmax, smax, vmax);
        inRange(hsv, lower, upper, mask);
        imshow("D", na_zywo);
        imshow("S", hsv);
        imshow("kolol", mask);
        waitKey(1);
    }*/
    //oko prawe
    Point  punkt_zlapany = Point( oko_prawe.x, oko_prawe.y);
    Point  punkt_przewidywany = punkt_zlapany;
    Point punkt_wlasciwy = punkt_zlapany;
    KalmanFilter filtr(4,2,0);
    filtr.transitionMatrix = (Mat_<float>(4,4) << 1, 0, 1, 0,  0,1,0,1, 0,0,1,0, 0,0,0,1);
    //cout << filtr.transitionMatrix; 
    //Inicjalizacja wktora stanu z fazy predykcji 
    filtr.statePre.at<float>(0) = oko_prawe.x;        //Położenie x 
    filtr.statePre.at<float>(1) = oko_prawe.y;        //Położenie y 
    filtr.statePre.at<float>(2) = 0;        //Prędkosc x 
    filtr.statePre.at<float>(3) = 0;        //Prędkosc y 
    //cout<<filtr.statePre;
    setIdentity (filtr.measurementMatrix);
    //cout << filtr.measurementMatrix; 
    setIdentity (filtr.processNoiseCov, Scalar::all(1e-4));
     //cout << filtr.processNoiseCov; 
    setIdentity(filtr.measurementNoiseCov, Scalar::all(1e-1)); 
    //cout << filtr.measurementNoiseCov; 
    setIdentity(filtr.errorCovPost, Scalar::all(.1)); 
     //cout << filtr.errorCovPost; 
     Mat_<float> macierz_pozycji(2,1);

    //oko Lewe
    Point  punkt_zlapany_l = Point( oko_lewe.x, oko_lewe.y);
    Point  punkt_przewidywany_l = punkt_zlapany_l;
    Point punkt_wlasciwy_l = punkt_zlapany_l;
    KalmanFilter filtr1(4,2,0);
    filtr1.transitionMatrix = (Mat_<float>(4,4) << 1, 0, 1, 0,  0,1,0,1, 0,0,1,0, 0,0,0,1);
    //cout << filtr.transitionMatrix; 
    //Inicjalizacja wktora stanu z fazy predykcji 
    filtr1.statePre.at<float>(0) = oko_lewe.x;        //Położenie x 
    filtr1.statePre.at<float>(1) = oko_lewe.y;        //Położenie y 
    filtr1.statePre.at<float>(2) = 0;        //Prędkosc x 
    filtr1.statePre.at<float>(3) = 0;        //Prędkosc y 
    //cout<<filtr.statePre;
    setIdentity (filtr1.measurementMatrix);
    //cout << filtr.measurementMatrix; 
    setIdentity (filtr1.processNoiseCov, Scalar::all(1e-4));
     //cout << filtr.processNoiseCov; 
    setIdentity(filtr1.measurementNoiseCov, Scalar::all(1e-1)); 
    //cout << filtr.measurementNoiseCov; 
    setIdentity(filtr1.errorCovPost, Scalar::all(.1)); 
     //cout << filtr.errorCovPost; 
     Mat_<float> macierz_pozycji_l(2,1);
     
     ofstream zapis("oko.txt");
    //Pętla główna
    while(waitKey(20)!=27)
    {   //sleep(0.5);
        //na_zywo=imread("oczy.jpg");
        przechwycenie>>na_zywo;
        //konwercja na szaro
        cvtColor(na_zywo, szare_zdjecie, COLOR_BGR2GRAY);
        imshow("szare zdjęcie",szare_zdjecie);
        //Znalezienie i zaznaczenie obszaru oczu
        kaskada_oczu.detectMultiScale(szare_zdjecie, oczy, 1.1,2,0|CASCADE_SCALE_IMAGE, Size(0,0));
        for(size_t i=0; i<oczy.size();i++)
        {
            kwadrat =oczy[i];
            // uciete.resize(oczy.size());
            //wycinek=oczy[i];
            //cout<<wycinek.x<<" "<<wycinek.y<<" "<<wycinek.width<<" "<<wycinek.height<<endl;
            rectangle(na_zywo,Point(kwadrat.x,kwadrat.y),Point(kwadrat.x+kwadrat.width-1,kwadrat.y+kwadrat.height-1), kolor1,3,8,0);
            //nie ciachamy
            // uciete[i]= na_zywo(oczy[i]);
        }
        /*for(int i=0; i<uciete.size(); i++)
        {    
            stringstream ss;
            ss << i;
            string str = ss.str();
            imshow(str, uciete[i]);
        }
        imshow("HJ", uciete[0]);
        imshow("d", uciete[1]);
        imshow("aaaa", uciete[1]);
        
        kaskada_oka_lewego.detectMultiScale(szare_zdjecie, oczy, 1.1,2,0|CASCADE_SCALE_IMAGE, Size(0,0));
        for(size_t i=0; i<oczy.size();i++)
        {
            kwadrat =oczy[i];
            wycinek=oczy[i];
            //cout<<wycinek.x<<" "<<wycinek.y<<" "<<wycinek.width<<" "<<wycinek.height<<endl;
            rectangle(na_zywo,Point(kwadrat.x,kwadrat.y),Point(kwadrat.x+kwadrat.width-1,kwadrat.y+kwadrat.height-1), kolor1,3,8,0);
        }
        
        //znalezienie twarzy i zaznaczenie jej
        cascade.detectMultiScale(szare_zdjecie, twarze, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(0,0));

        for(size_t i=0; i<twarze.size(); i++)
        {
            kwadrat =twarze[i];
            rectangle(na_zywo,Point(kwadrat.x,kwadrat.y),Point(kwadrat.x+kwadrat.width-1,kwadrat.y+kwadrat.height-1), kolor,3,8,0);
        }
        
        
        //źrenice
        //znalezienie czarnego obszaru
        cvtColor(na_zywo,hsv, COLOR_BGR2HSV);
        // imshow("HSV", hsv);
        Scalar lower(hmin, smin, vmin);
        Scalar upper (hmax, smax, vmax);
        inRange(hsv, lower, upper, mask);
        // imshow ("OCZY", mask);
        
        //dylatacja i erozja do wypełnienia ewentualnych dziur wewnątrz okręgu
        Mat kernel =getStructuringElement(MORPH_RECT, Size(7,7));
        dilate(mask, dylatacja, kernel) ;
        Mat kernel1 =getStructuringElement(MORPH_RECT, Size(5,5));
        erode(dylatacja, erozja, kernel1);
       // imshow ("erozja", erozja);
        //Zaznaczenie krawędzi
        GaussianBlur(erozja,Blur, Size(5,5),3,0);
        Canny(Blur,kontury,25,70);
        Mat kernel2=getStructuringElement(MORPH_RECT, Size(3,3));
        dilate(kontury, kontury,kernel2);
        //imshow("kontury", kontury);
        
        co tu sie dzieje
        vector<vector<Point>> znalezione;
        vector<Vec4i> hierarchy;
        findContours(kontury,znalezione,hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for( int i=0; i<znalezione.size(); i++)
        {
            Rect area=boundingRect(znalezione[i]);
            for( int j=0; j<oczy.size(); j++)
            { 
                if(area.x>oczy[j].x && area.x<oczy[j].x+oczy[j].width &&area.y>oczy[j].y && area.y < oczy[j].y+oczy[j].height)
                {
                    drawContours(konturki,znalezione, i, Scalar(255,0,255),10);
                    drawContours(na_zywo,znalezione, i, Scalar(255,0,255),10);

                }
            }
        }
        HoughLines(kontury,linie,1,CV_PI/180, 150,50,50);
         Mat imgBlur;
        medianBlur();
        
        Mat img =imread("zdjecie.jpg", IMREAD_COLOR);

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        vector <Vec3f> circles;
        HoughCircles(gray,circles,HOUGH_GRADIENT, 1,kontury.rows/64,200,10,5,30);
        
        for(size_t i=0; i<circles.size();i++)
        {
           // Vec3i c=linie[i];
            Point srodek=Point(cvRound(circles[i][0]),cvRound(circles[i][1]));

            
            int kat=cvRound(circles[i][2]);
            //circle(na_zywo,srodek, 1, Scalar(255,255,255), FILLED);
            circle(img,srodek, kat, Scalar(255,0,255),2, 8, 0);
            
        }*/
        

	    // Read the image as gray-scale
    	Mat img = na_zywo;
	    // Convert to gray-scale
	    Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
	    // Blur the image to reduce noise
	    Mat img_blur;
	    medianBlur(gray, img_blur, 5);
	    // Create a vector for detected circles
	    vector<Vec3f>  circles;
	    // Apply Hough Transform
	    HoughCircles(img_blur, circles, HOUGH_GRADIENT, 1, img.cols/8, 150, 10, 8, 10);
    	// Draw detected circles
        //vector <Point> srodek_oka;
        oko_lewe=Point(0,0);
        oko_prawe=Point(0,0); 
	    Point &srodek_oka_lewego=oko_lewe;
        Point &srodek_oka_prawego=oko_prawe; 
        przefiltrowane_x=Point(0,0);
        float srodek;
        if(!oczy.empty())
            srodek=oczy[0].x+oczy[0].width/2;
    
        for(size_t i=0; i<circles.size(); i++) 
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            //Rect area=boundingRect(circles[i]);
            for( int j=0; j<oczy.size(); j++)
            {
                if(center.x>oczy[j].x && center.x<oczy[j].x+oczy[j].width &&center.y>oczy[j].y && center.y < oczy[j].y+oczy[j].height)
                {
                    vector <bool> czylewe(2);
                    if(oczy[0].x<oczy[1].x)
                    {   czylewe[1]=false;
                        czylewe[0]=true;
                    }
                    else
                    {   czylewe[1]=true;
                        czylewe[0]=false;
                    }
                    
                    int radius = cvRound(circles[i][2]);
                    circle(img, center, radius, Scalar(255, 255, 255), 2, 8, 0);
                    if(czylewe[j])
                    {
        	            srodek_oka_lewego=center;
                        macierz_pozycji_l(0) = oko_lewe.x;
                        macierz_pozycji_l(1) = oko_lewe.y;
                        filtr1.predict();
                        Mat przewidziane_l= filtr1.correct(macierz_pozycji_l);
                        Point przewidziany_punkt_l (przewidziane_l.at<float>(0), przewidziane_l.at<float>(1));
                        blokada.lock();
                        przefiltrowane_l=przewidziany_punkt_l;
                        zapis<<"x "<<srodek_oka_lewego.x<<" ";
                        zapis<<"y "<<srodek_oka_lewego.y<<endl;
                        zapis<<"przef x "<<przefiltrowane_l.x<<" ";
                        zapis<<"przef y "<<przefiltrowane_l.y<<endl;
                        circle(img, przefiltrowane_l, radius, Scalar(255, 0, 0), 2, 8, 0);
                        blokada.unlock();
                    }
                    else
                    {
                        srodek_oka_prawego=center;
                        macierz_pozycji(0) = oko_prawe.x;
                        macierz_pozycji(1) = oko_prawe.y;
                        filtr.predict();
                        Mat przewidziane= filtr.correct(macierz_pozycji);
                        Point przewidziany_punkt (przewidziane.at<float>(0), przewidziane.at<float>(1));
                        blokada.lock();
                        przefiltrowane_x=przewidziany_punkt;
                        zapis<<"x "<<srodek_oka_prawego.x<<" ";
                        zapis<<"y "<<srodek_oka_prawego.y<<endl;
                        zapis<<"przef x "<<przefiltrowane_x.x<<" ";
                        zapis<<"przef y "<<przefiltrowane_x.y<<endl;
                        circle(img, przefiltrowane_x, radius, Scalar(255, 0, 0), 2, 8, 0);
                        blokada.unlock();
                    }


                }
            }  
            
        }
        //  CZY MOGE JJUZ ISC DALEJ XD  
        /*
          cout<<oko_prawe.x<<"  "<<oko_lewe.y<<"           "<<oko_lewe.x<<"  "<<oko_lewe.y<<endl;
            if(srodek_oka_prawego.x==0 && srodek_oka_lewego.x==0)
        {
            t++;
            if(t>20)
                cout<<"Mrugniecie";
        }
        else
        {
            t=0;
        }
        //HoughCircles(kontury,na_zywo,HOUGH_GRADIENT,1, kontury.rows/20,100, 30, 0, 30);
            if(mrugniecie())
            cout<<" MRUGNIECIE AAAAAAA";
       
        macierz_pozycji(0) = oko_prawe.x;
        macierz_pozycji(1) = oko_prawe.y;
        filtr.predict();
        Mat przewidziane= filtr.correct(macierz_pozycji);
        Point przewidziany_punkt (przewidziane.at<float>(0), przewidziane.at<float>(1));
        przefiltrowane_x=przewidziany_punkt;
        zapis<<"x "<<srodek_oka_prawego.x<<" ";
        zapis<<"y "<<srodek_oka_prawego.y<<endl;
        zapis<<"przef x "<<przefiltrowane_x.x<<" ";
        zapis<<"przef y "<<przefiltrowane_x.y<<endl;*/
        if (na_zywo.empty())
            cout<<"k";
        imshow("live",na_zywo);

    }
    zapis.close();
}
bool mrugniecie(Point p)
{
      //cout<<p.x<<"  "<<p.y<<endl;

    if(p.x==0 && p.x==0)
        {
            t++;
            if(t>10)
                return true;

            //cout<<t<<" | ";
            return false;
        }
    else
        {
            t=0;
            return false;
        }
}

/////////////////////////////////////////
void f2()
{
    Point kopia, kopial;
    Display* d=XOpenDisplay(NULL);
    Screen* s= DefaultScreenOfDisplay(d);
    //int scr= XDefaultScreen(d);
    //Window okno = XRootWindow(d, scr);
    ruch_poziomy[0]=0;
    ruch_pionowy[0]=0;
    ruch_poziomy[1]=0;
    ruch_pionowy[1]=0;

    Point poprzedni=Point(0,0);
    sleep(10);
    vector <Point> punkty(5);
    Point tymczasowy;
    fill(punkty.begin(),punkty.end(), Point(0,0));
    Mat konfiguracja(s->height,s->width,CV_8UC3, Scalar(0,0,0));
    namedWindow ("Konfiguracja", WINDOW_NORMAL);
    setWindowProperty("Konfiguracja", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    while(punkty[4].x==0)
    {   
        blokada.lock();
        kopia.x=przefiltrowane_x.x;
        kopia.y=przefiltrowane_x.y;
        blokada.unlock();
        if(kopia.x!=0 && kopia.y!=0)
        { //bool a=true;
                if(poprzedni!=Point(0,0))
                {
                        poprzedni=tymczasowy;
                        tymczasowy=kopia;
                }
                else
                {
                    tymczasowy =kopia;
                    poprzedni=tymczasowy;
                }  
        }
        if(punkty[3].x!=0 && punkty[4].x==0)
        {
            
            circle(konfiguracja, Point(s->width/2,s->height/2), 50, Scalar(0,155,0), FILLED);
            
            if(mrugniecie(kopia)) 
            {
                punkty[4].x=tymczasowy.x;
                punkty[4].y=tymczasowy.y;
                circle(konfiguracja, Point(s->width/2,s->height/2), 50, Scalar(0,0,0), FILLED);
                sleep(1);
            }
        }
        if(punkty[2].x!=0 && punkty[3].x==0)
        {
            circle(konfiguracja, Point(50,s->height-50), 50, Scalar(0,155,0), FILLED);
           
            if(mrugniecie(kopia))
            {   
                punkty[3].x=tymczasowy.x;
                punkty[3].y=tymczasowy.y;
                circle(konfiguracja, Point(50,s->height-50), 50, Scalar(0,0,0), FILLED);
                sleep(3);
            }
        }
        if(punkty[1].x!=0 &&punkty[2].x==0)
        {
            circle(konfiguracja, Point(s->width-50,s->height-50), 50, Scalar(0,155,0), FILLED);
           
            if(mrugniecie(kopia))
            {
                punkty[2].x=tymczasowy.x;
                punkty[2].y=tymczasowy.y;
                circle(konfiguracja, Point(s->width-50,s->height-50), 50, Scalar(0,0,0), FILLED);
                sleep(3);
            }
        }
        if(punkty[0].x!=0 && punkty[1].x==0)
        {
            circle(konfiguracja, Point(s->width-50,50), 50, Scalar(0,155,0), FILLED);
            
            if(mrugniecie(kopia))
            {   
                punkty[1].x=tymczasowy.x;
                punkty[1].y=tymczasowy.y;
                circle(konfiguracja, Point(s->width-50,50), 50, Scalar(0,0,0), FILLED);
                sleep(3); 
            }
        }
        if(punkty[0].x==0)
        {
            circle(konfiguracja, Point(50,50), 50, Scalar(250,0,125), FILLED);
            
            if(mrugniecie(kopia))
            {
                punkty[0].x=tymczasowy.x;
                punkty[0].y=tymczasowy.y;
                circle(konfiguracja, Point(50,50), 50, Scalar(0,0,0), FILLED);
            }
        }
        imshow ("Konfiguracja", konfiguracja);
    }
    //cout<<"HE";
    destroyWindow("Konfiguracja");
    srodek=punkty[4];
   // ruch_poziomy=(punkty[1].x-punkty[0].x+punkty[2].x-punkty[3].x)/2/450;
    //ruch_pionowy=(punkty[3].y-punkty[0].y+punkty[2].y-punkty[1].y)/2/450;
    Mat sledzenie(s->height,s->width,CV_8UC3, Scalar(0,0,0));
     namedWindow ("sledzenie", WINDOW_NORMAL);
    setWindowProperty("sledzenie", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    while(true)
    {
        blokada.lock();
        kopia.x=przefiltrowane_x.x;
        kopia.y=przefiltrowane_x.y;
        blokada.unlock();
       if(kopia.x!=0 && kopia.y!=0 )//&& oko_lewe.x!=0 && oko_lewe.y!=0)
        {
            ruch_poziomy[1]=punkty[4].x*s->width/2/(kopia.x);
            ruch_pionowy[1]=punkty[4].y*s->height/2/(kopia.y);
            circle(sledzenie, Point(ruch_poziomy[0],ruch_pionowy[0]), 50, Scalar(0,0,0), FILLED);
            circle(sledzenie, Point(ruch_poziomy[1],ruch_pionowy[1]), 50, Scalar(255,0,0), FILLED);
            // XWarpPointer(d, None, okno, 0, 0, 0, 0, ruch_poziomy[1], ruch_pionowy[1]);
           // XFlush(d);
        }
        ruch_poziomy[0]=ruch_poziomy[1];
        ruch_pionowy[0]=ruch_pionowy[1];
        imshow("sledzenie", sledzenie);
    }
}  
int main()
{
    //int n = 0;
    std::thread thd1(f1); 
    thread thd2(f2); 
    thd1.join();
    thd2.join();
    //std::cout << "working" <<std::endl;
    return 0;
}
// zrób mrugniecie dla lewego oka tez ogolnie cały f2 zrób kalmana dla drugiego oka 
//i powinno sie naprawić żeby po wyzerowanych okch nie rzucało
