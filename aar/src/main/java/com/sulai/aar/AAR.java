package com.sulai.aar;

import android.content.Context;
import android.util.Log;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by sascha wernegger on 17.08.2016.
 */
public class AAR {


    private static final String TAG = "SPHERO_SMILES";
    public static AAR obj;

    private static Context context;

    private static BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(AAR.context) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    Toast.makeText(AAR.context,"OpenCV loaded successfully",Toast.LENGTH_LONG).show();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public static void  setContext(Context context){
        AAR.context=context;
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0,context,AAR.mLoaderCallback);
    }
    public static AAR instance(){
        if(obj==null)
            obj = new AAR();
        return obj;
    }

    public float[] track(byte[] data, int width, int height){
        Mat m = new Mat(width,height, CvType.CV_8UC3);
        m.put(0,0,data);
        Imgproc.cvtColor(m,m,Imgproc.COLOR_RGB2GRAY);
        Imgproc.resize(m,m,new Size(m.size().width/2,m.size().height/2));
        Imgproc.threshold(m,m,200,250,Imgproc.THRESH_BINARY);
        Imgproc.blur(m,m,new Size(6,6));
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(m,contours,hierarchy, Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE);
        if(contours.isEmpty())
            return new float[]{1f,1f,1f};
        Point[] points = contours.get(0).toArray();
        MatOfPoint2f contour = new MatOfPoint2f();
        contour.fromArray(points);
        Point center= new Point();
        float[] radii = new float[1];

        Imgproc.minEnclosingCircle(contour,center ,radii);
        return new float[]{(float)center.x, (float)center.y, radii[0]};
    }

    public byte[] trackDBG(byte[] data, int width, int height){
        Mat m = new Mat(width,height, CvType.CV_8UC3);
        m.put(0,0,data);
        Imgproc.cvtColor(m,m,Imgproc.COLOR_RGB2GRAY);
        Imgproc.threshold(m,m,200,250,Imgproc.THRESH_BINARY);
        Imgproc.blur(m,m,new Size(6,6));
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(m,contours,hierarchy, Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE);
        if(contours.isEmpty())
            return getBytes(m);
        Point[] points = contours.get(0).toArray();
        MatOfPoint2f contour = new MatOfPoint2f();
        contour.fromArray(points);
        Point center= new Point();
        float[] radii = new float[1];

        Imgproc.minEnclosingCircle(contour,center ,radii);
        Imgproc.drawContours(m,contours,-1,new Scalar(255,255,255));
        Imgproc.circle(m,center, (int) radii[0],new Scalar(255,0,0));
        return getBytes(m);
    }

    private byte[] getBytes(Mat m) {
        byte[] data = new byte[m.rows() * m.cols()];
        m.get(0,0,data);
        return data;
    }
}
