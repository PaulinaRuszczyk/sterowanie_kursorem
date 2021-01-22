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
int pom;
Point oko_lewe=Point(0,0);
Point oko_prawe=Point(0,0); 
Point srodek, srodekl;
Point  przefiltrowane_x;
Point przefiltrowane_l;
int t=0;
vector <int> ruch_poziomy(2);
vector <int> ruch_pionowy(2);
//ruch_poziomy[0]=0;
//ruch_pionowy[0]=0;

//parametry do znalezeinia koloru
int hmax=255;
int hmin=0;
int smax=255;
int smin=0;
int vmax=70;
int vmin=0;
//macierze
Mat na_zywo, dylatacja, erozja, Blur;
Mat rysuj;
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
    kaskada_oczu.load("haarcascade_eye.xml.2");
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
        //equalizeHist(hsv, hsv);
        Scalar lower(hmin, smin, vmin);
        Scalar upper (hmax, smax, vmax);
        inRange(hsv, lower, upper, mask);
        imshow("D", na_zywo);
        imshow("S", hsv);
        imshow("kolol", mask);
        waitKey(1);
    }
    */
    // Filtr Kalmana

    //oko prawe
    Point  punkt_zlapany = Point( oko_prawe.x, oko_prawe.y);
    Point  punkt_przewidywany = punkt_zlapany;
    Point punkt_wlasciwy = punkt_zlapany;
    KalmanFilter filtr(4,2,0);
    filtr.transitionMatrix = (Mat_<float>(4,4) << 1, 0, 1, 0,  0,1,0,1, 0,0,1,0, 0,0,0,1);
    //Inicjalizacja wktora stanu z fazy predykcji 
    filtr.statePre.at<float>(0) = oko_prawe.x;        //Położenie x 
    filtr.statePre.at<float>(1) = oko_prawe.y;        //Położenie y 
    filtr.statePre.at<float>(2) = 0;                  //Prędkosc x 
    filtr.statePre.at<float>(3) = 0;                  //Prędkosc y 
    setIdentity (filtr.measurementMatrix);
    setIdentity (filtr.processNoiseCov, Scalar::all(1e-4));
    setIdentity(filtr.measurementNoiseCov, Scalar::all(1e-1)); 
    setIdentity(filtr.errorCovPost, Scalar::all(.1)); 
     Mat_<float> macierz_pozycji(2,1);

    //oko Lewe
    Point  punkt_zlapany_l = Point( oko_lewe.x, oko_lewe.y);
    Point  punkt_przewidywany_l = punkt_zlapany_l;
    Point punkt_wlasciwy_l = punkt_zlapany_l;
    KalmanFilter filtr1(4,2,0);
    filtr1.transitionMatrix = (Mat_<float>(4,4) << 1, 0, 1, 0,  0,1,0,1, 0,0,1,0, 0,0,0,1);
    //Inicjalizacja wktora stanu z fazy predykcji 
    filtr1.statePre.at<float>(0) = oko_lewe.x;        //Położenie x 
    filtr1.statePre.at<float>(1) = oko_lewe.y;        //Położenie y 
    filtr1.statePre.at<float>(2) = 0;        //Prędkosc x 
    filtr1.statePre.at<float>(3) = 0;        //Prędkosc y 
    setIdentity (filtr1.measurementMatrix);
    setIdentity (filtr1.processNoiseCov, Scalar::all(1e-4));
    setIdentity(filtr1.measurementNoiseCov, Scalar::all(1e-1)); 
    setIdentity(filtr1.errorCovPost, Scalar::all(.1)); 
     Mat_<float> macierz_pozycji_l(2,1);
     
     //ofstream zapis("oko.txt");

    //Pętla główna
    while(waitKey(20)!=27)
    {   
        przechwycenie>>na_zywo;

       imwrite("nazywo.jpg", na_zywo);
       na_zywo=imread("nazywo.jpg",IMREAD_COLOR);
        //konwersja na szaro
        cvtColor(na_zywo, szare_zdjecie, COLOR_RGB2GRAY);
        imshow("szare zdjęcie",szare_zdjecie);

        //equalizeHist(szare_zdjecie, szare_zdjecie);

        // Znalezienie i zaznaczenie obszaru oczu
        cascade.detectMultiScale(szare_zdjecie, twarze, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(0,0));
        for(size_t i=0; i<twarze.size(); i++)
        {
            kwadrat =twarze[i];
            rectangle(na_zywo,Point(kwadrat.x,kwadrat.y),Point(kwadrat.x+kwadrat.width-1,kwadrat.y+kwadrat.height-1), kolor,3,8,0);
        }
        //Znalezienie i zaznaczenie obszaru oczu
        kaskada_oczu.detectMultiScale(szare_zdjecie, oczy, 1.1,2,0|CASCADE_SCALE_IMAGE, Size(0,0));
        for(size_t i=0; i<oczy.size();i++)
        {
            
            kwadrat =oczy[i];
            for(size_t j=0; j<twarze.size(); j++)
            {
                if(kwadrat.x>twarze[j].x+10 && kwadrat.x<twarze[j].x+twarze[j].width-10 && kwadrat.y>twarze[j].y+10 && kwadrat.y<twarze[j].y+twarze[j].height-50)
                rectangle(na_zywo,Point(kwadrat.x,kwadrat.y),Point(kwadrat.x+kwadrat.width-1,kwadrat.y+kwadrat.height-1), kolor1,3,8,0);
            }
        }
        
        //źrenice metodą znajdowania koloru

        //znalezienie czarnego obszaru
        cvtColor(na_zywo,hsv, COLOR_BGR2HSV);
         imshow("HSV", hsv);
        Scalar lower(hmin, smin, vmin);
        Scalar upper (hmax, smax, vmax);
        inRange(hsv, lower, upper, mask);
         imshow ("OCZY", mask);
        
        //dylatacja i erozja do wypełnienia ewentualnych dziur wewnątrz okręgu
        Mat kernel1 =getStructuringElement(MORPH_RECT, Size(4,4));
        Mat kernel =getStructuringElement(MORPH_RECT, Size(3,3));
        dilate(mask, dylatacja, kernel);
        for(int i=0; i<3; i++)
        {
            erode(dylatacja, erozja, kernel1);
            dilate(erozja, dylatacja, kernel) ;
        }
        dilate(dylatacja, dylatacja, kernel1);
        imshow("dylatacja", dylatacja);

        // wyciecie obszaru oczu  z obrazu binarnego
        vector <Mat> binarny;
        for(int i=0; i<oczy.size();i++)
        {
            binarny.resize(oczy.size());
            binarny[i]=dylatacja(oczy[i]);
            char a=i;
            imshow("a", binarny[i]);
        }

        vector <Point> punkty;
        punkty.resize(binarny.size());
        Point srodek_lewy;
        Point srodek_prawy;
        int sumax, sumay, ilosc_px;
     /* oko_lewe=Point(0,0);
        oko_prawe=Point(0,0); 
	    Point &srodek_oka_lewego=oko_lewe;
        Point &srodek_oka_prawego=oko_prawe; 
/*
        //trzeba zrobić zmienną żeby było oko 

        //Znalezienie źrenic licząć środek jedynek
        for(int i=0; i<binarny.size(); i++)
        {   sumax=0;
            sumay=0;
            ilosc_px=0;
            for (int r=0 ; r<binarny[i].rows; r++)
            {
                Vec3b* ptr = binarny[i].ptr<cv::Vec3b>(r); 
                for (int c = 0 ; c < binarny[i].cols ; c++)
                {
                    if(ptr[c] == Vec3b(255, 255, 255))
                    {
                        sumax+=c;
                        sumay+=r;
                        ilosc_px++;
                    }
                    
                }
            }
            if(ilosc_px!=0)
            {
                punkty[i].x=sumax/ilosc_px;
                punkty[i].y=sumay/ilosc_px;
                circle(na_zywo, Point(punkty[i].x+oczy[i].x,punkty[i].y+oczy[i].y), 15, Scalar(0,0,255), 4);
            }
            else
            {
                punkty[i].x=0;
                punkty[i].y=0;
            }
            cout<<punkty[i].x<<"   "<<punkty[i].y<< "|";
            if(i==1)
            {
                vector <bool> czylewe(2);
                if(punkty[0].x+oczy[0].x<punkty[1].x+oczy[1].x)
                {   czylewe[1]=false;
                    czylewe[0]=true;
                }
                else
                {   czylewe[1]=true;
                    czylewe[0]=false;
                }
                for(int j=0; j<2; j++)
                {
                    if(czylewe[j])
                    {
        	            srodek_oka_lewego=Point(punkty[j].x+oczy[j].x, punkty[j].y+oczy[j].y);
                        macierz_pozycji_l(0) = oko_lewe.x;
                        macierz_pozycji_l(1) = oko_lewe.y;
                        filtr1.predict();
                        Mat przewidziane_l= filtr1.correct(macierz_pozycji_l);
                        Point przewidziany_punkt_l (przewidziane_l.at<float>(0), przewidziane_l.at<float>(1));
                        blokada.lock();
                        przefiltrowane_l=przewidziany_punkt_l;
                        // zapis<<"x "<<srodek_oka_lewego.x<<" ";
                        // zapis<<"y "<<srodek_oka_lewego.y<<endl;
                        // zapis<<"przef x "<<przefiltrowane_l.x<<" ";
                        // zapis<<"przef y "<<przefiltrowane_l.y<<endl;
                        circle(na_zywo, przefiltrowane_l, 15, Scalar(255, 255, 0), 2, 8, 0);
                        blokada.unlock();
                        
                    }
                    else
                    {
                        srodek_oka_prawego=Point(punkty[j].x+oczy[j].x, punkty[j].y+oczy[j].y);
                        macierz_pozycji(0) = oko_prawe.x;
                        macierz_pozycji(1) = oko_prawe.y;
                        filtr.predict();
                        Mat przewidziane= filtr.correct(macierz_pozycji);
                        Point przewidziany_punkt (przewidziane.at<float>(0), przewidziane.at<float>(1));
                        blokada.lock();
                        przefiltrowane_x=przewidziany_punkt;
                        // zapis<<"x "<<srodek_oka_prawego.x<<" ";
                        // zapis<<"y "<<srodek_oka_prawego.y<<endl;
                        // zapis<<"przef x "<<przefiltrowane_x.x<<" ";
                        // zapis<<"przef y "<<przefiltrowane_x.y<<endl;
                        circle(na_zywo, przefiltrowane_x, 15, Scalar(255, 255, 0), 2, 8, 0);
                        blokada.unlock();
                    }


                }
            }
        }
*/
       // Zaznaczenie krawędzi Cannym 
        GaussianBlur(mask,Blur, Size(5,5),3,0);
        Canny(Blur,kontury,25,70);
        Mat kernel2=getStructuringElement(MORPH_RECT, Size(3,3));
        dilate(kontury, kontury,kernel2);
        //imshow("kontury", kontury);

       // iny sposob
       Mat src_szary;
       kontury.convertTo(rysuj, CV_8U);
        imshow("coto", rysuj);
        imwrite("output.jpg", rysuj);
        Mat src= imread("output.jpg",1);
        cvtColor(src, src_szary, COLOR_RGB2GRAY);
        GaussianBlur(src_szary, src_szary, Size(9,9),2 ,2);
        vector <Vec3f> circles;
        HoughCircles(src_szary, circles, HOUGH_GRADIENT, 2, src_szary.rows / 8, 150, 10, 8, 20);
        oko_lewe=Point(0,0);
        oko_prawe=Point(0,0); 
	    Point &srodek_oka_lewego=oko_lewe;
        Point &srodek_oka_prawego=oko_prawe; 
        przefiltrowane_x=Point(0,0);
        przefiltrowane_l=Point(0,0);
        for(size_t i=0; i<circles.size(); i++) 
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
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
                    circle(na_zywo, center, radius, Scalar(255, 255, 255), 2, 8, 0);
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
                        // zapis<<"x "<<srodek_oka_lewego.x<<" ";
                        // zapis<<"y "<<srodek_oka_lewego.y<<endl;
                        // zapis<<"przef x "<<przefiltrowane_l.x<<" ";
                        // zapis<<"przef y "<<przefiltrowane_l.y<<endl;
                        circle(na_zywo, przefiltrowane_l, radius, Scalar(255, 0, 0), 2, 8, 0);
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
                        // zapis<<"x "<<srodek_oka_prawego.x<<" ";
                        // zapis<<"y "<<srodek_oka_prawego.y<<endl;
                        // zapis<<"przef x "<<przefiltrowane_x.x<<" ";
                        // zapis<<"przef y "<<przefiltrowane_x.y<<endl;
                        circle(na_zywo, przefiltrowane_x, radius, Scalar(255, 0, 0), 2, 8, 0);
                        blokada.unlock();
                    }


                }
            }  
            
        }
        imshow("CZy", src);

        /*
      //  znajdowanie konturów za pomocą find contours
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
        */
        
/*
	    // znalezienie oczy za pomocą hough circles
	    Mat img_blur;
	    medianBlur(szare_zdjecie, img_blur, 5);
	    vector<Vec3f>  circles;
        equalizeHist(img_blur, img_blur);
	    HoughCircles(img_blur, circles, HOUGH_GRADIENT, 1, na_zywo.cols/8, 150, 10, 8, 10);
        oko_lewe=Point(0,0);
        oko_prawe=Point(0,0); 
	    Point &srodek_oka_lewego=oko_lewe;
        Point &srodek_oka_prawego=oko_prawe; 
        przefiltrowane_x=Point(0,0);
        przefiltrowane_l=Point(0,0);
        for(size_t i=0; i<circles.size(); i++) 
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
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
                    circle(na_zywo, center, radius, Scalar(255, 255, 255), 2, 8, 0);
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
                        circle(na_zywo, przefiltrowane_l, radius, Scalar(255, 0, 0), 2, 8, 0);
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
                        circle(na_zywo, przefiltrowane_x, radius, Scalar(255, 0, 0), 2, 8, 0);
                        blokada.unlock();
                    }


                }
            }  
            
        }*/
        //Filtr Kalmana przewidywanie i poprawa
        //  macierz_pozycji(0) = oko_prawe.x;
        // macierz_pozycji(1) = oko_prawe.y;
        // filtr.predict();
        // Mat przewidziane= filtr.correct(macierz_pozycji);
        // Point przewidziany_punkt (przewidziane.at<float>(0), przewidziane.at<float>(1));
        // przefiltrowane_x=przewidziany_punkt;
        // zapis<<"x "<<srodek_oka_prawego.x<<" ";
        // zapis<<"y "<<srodek_oka_prawego.y<<endl;
        // zapis<<"przef x "<<przefiltrowane_x.x<<" ";
        // zapis<<"przef y "<<przefiltrowane_x.y<<endl;
       // cout<<endl;
        if (na_zywo.empty())
            cout<<"k";
        
        imshow("live",na_zywo);

    }
    //zapis.close();
}

//Funkcja wykrywająca mrugnięcie
bool mrugniecie(Point p, Point l)
{
     //cout<<p.x<<"  "<<p.y<<"|  " <<l.x<<"      "<<l.y<<endl;

    if(p.x==0 && p.y==0 &&l.x==0 && l.y==0)
    {

        if(t>3)
        {
            t=0;
            return true;
        }
        t++;
        cout<<t<<" | ";
        return false;
    }
    else
    {
        t=0;
        return false;
    }
}
//Drugi wątek do konfiguracji i wyświetlenia okna ze śledzonym wzrokiem
void f2()
{
    Point kopia, kopial;
    Display* d=XOpenDisplay(NULL);
    Screen* s= DefaultScreenOfDisplay(d);

    ruch_poziomy[0]=0;
    ruch_pionowy[0]=0;
    ruch_poziomy[1]=0;
    ruch_pionowy[1]=0;

    Point poprzedni=Point(0,0);
    Point poprzednil=Point(0,0);
    sleep(10);
    vector <Point> punkty(5);
    vector <Point> punktyl(5);

    Point tymczasowy, tymczasowyl;
    fill(punkty.begin(),punkty.end(), Point(0,0));
    fill(punktyl.begin(),punktyl.end(), Point(0,0));
    Mat konfiguracja(s->height,s->width,CV_8UC3, Scalar(0,0,0));
    namedWindow ("Konfiguracja", WINDOW_NORMAL);
    setWindowProperty("Konfiguracja", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    //Wyświetlenie okna konfiguracji i zebranie punktów pomiarowych
    while(punkty[4].x==0)
    {   
        blokada.lock();
        kopia.x=przefiltrowane_x.x;
        kopia.y=przefiltrowane_x.y;
        kopial.x=przefiltrowane_l.x;
        kopial.y=przefiltrowane_l.y;
        blokada.unlock();
        if(kopia.x!=0 && kopia.y!=0 && kopial.x!=0 && kopial.y!=0)
        { 
                if(poprzedni!=Point(0,0))
                {
                        poprzedni=tymczasowy;
                        tymczasowy=kopia;
                        poprzednil=tymczasowyl;
                        tymczasowyl=kopial;
                }
                else
                {
                    tymczasowy =kopia;
                    poprzedni=tymczasowy;
                    tymczasowyl =kopial;
                    poprzednil=tymczasowyl;
                }  
        }
        if(punkty[3].x!=0 && punkty[4].x==0)
        {
            circle(konfiguracja, Point(s->width/2,s->height/2), 50, Scalar(0,155,0), FILLED);
            
            if(mrugniecie(kopia, kopial)) 
            {
                punkty[4].x=tymczasowy.x;
                punkty[4].y=tymczasowy.y;
                punktyl[4].x=tymczasowyl.x;
                punktyl[4].y=tymczasowyl.y;
                circle(konfiguracja, Point(s->width/2,s->height/2), 50, Scalar(0,0,0), FILLED);
                
            }
        }
        if(punkty[2].x!=0 && punkty[3].x==0)
        {
            circle(konfiguracja, Point(50,s->height-50), 50, Scalar(0,155,0), FILLED);
           
            if(mrugniecie(kopia, kopial))
            {   
                punkty[3].x=tymczasowy.x;
                punkty[3].y=tymczasowy.y;
                punktyl[3].x=tymczasowyl.x;
                punktyl[3].y=tymczasowyl.y;
                circle(konfiguracja, Point(50,s->height-50), 50, Scalar(0,0,0), FILLED);
            }
        }
        if(punkty[1].x!=0 &&punkty[2].x==0)
        {
            circle(konfiguracja, Point(s->width-50,s->height-50), 50, Scalar(0,155,0), FILLED);
           
            if(mrugniecie(kopia, kopial))
            {
                punkty[2].x=tymczasowy.x;
                punkty[2].y=tymczasowy.y;
                punktyl[2].x=tymczasowyl.x;
                punktyl[2].y=tymczasowyl.y;
                circle(konfiguracja, Point(s->width-50,s->height-50), 50, Scalar(0,0,0), FILLED);
                sleep(3);
            }
        }
        if(punkty[0].x!=0 && punkty[1].x==0)
        {
            circle(konfiguracja, Point(s->width-50,50), 50, Scalar(0,155,0), FILLED);
            if(mrugniecie(kopia, kopial))
            {   
                punkty[1].x=tymczasowy.x;
                punkty[1].y=tymczasowy.y;
                punktyl[1].x=tymczasowyl.x;
                punktyl[1].y=tymczasowyl.y;
                circle(konfiguracja, Point(s->width-50,50), 50, Scalar(0,0,0), FILLED);
            }
        }
        if(punkty[0].x==0)
        {
            circle(konfiguracja, Point(50,50), 50, Scalar(250,0,125), FILLED);
            
            if(mrugniecie(kopia, kopial))
            {
                punkty[0].x=tymczasowy.x;
                punkty[0].y=tymczasowy.y;
                punktyl[0].x=tymczasowyl.x;
                punktyl[0].y=tymczasowyl.y;
                circle(konfiguracja, Point(50,50), 50, Scalar(0,0,0), FILLED);
            }
        }
        imshow ("Konfiguracja", konfiguracja);
    }
    destroyWindow("Konfiguracja");


    srodek=punkty[4];
    //Pokazanie śledzonego wzroku
    Mat sledzenie(s->height,s->width,CV_8UC3, Scalar(0,0,0));
    namedWindow ("sledzenie", WINDOW_NORMAL);
    setWindowProperty("sledzenie", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    while(true)
    {
        blokada.lock();
        kopia.x=przefiltrowane_x.x;
        kopia.y=przefiltrowane_x.y;

        blokada.unlock();
       if(kopia.x!=0 && kopia.y!=0 )
        {
            ruch_poziomy[1]=punkty[4].x*s->width/2/(kopia.x);
            ruch_pionowy[1]=punkty[4].y*s->height/2/(kopia.y);
            circle(sledzenie, Point(ruch_poziomy[0],ruch_pionowy[0]), 50, Scalar(0,0,0), FILLED);
            circle(sledzenie, Point(ruch_poziomy[1],ruch_pionowy[1]), 50, Scalar(255,0,0), FILLED);
            //Ruch myszą
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
    std::thread thd1(f1); 
    thread thd2(f2); 
    thd1.join();
    thd2.join();
    return 0;
}
// zrobic stopien wychylenia gdzzie początek układu współrzędnych byłby w lewym górnym rogu
// natepnie mape jaki stopien w jakim polozeniu