package org.opencv.samples.tutorial1.opencv;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by sascha wernegger on 10.08.2016.
 */
public class ObjTracker {

    private int bins = 255;

    private FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
    private DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
    private DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
    private int k = 10;
    private boolean bCalibrated = false;
    private int channel = 2;

    public Mat calibrate(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        bCalibrated = true;
        Mat channel = getHSVChannels(inputFrame.rgba());
        Mat desc = calcDescriptors(channel);
        return channel;
    }

    private Mat getHSVChannels(Mat rgba) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(rgba,hsv,Imgproc.COLOR_RGB2HSV);
        List<Mat> channels = new ArrayList<>();
        Core.split(hsv,channels);
        return channels.get(channel);
    }

    private Mat calcDescriptors(Mat gray) {
        Mat desc = new Mat();
        MatOfKeyPoint kp = new MatOfKeyPoint();
        detector.detect(gray,kp);
        extractor.compute(gray,kp,desc);
        ArrayList<Mat> descArray = new ArrayList<>();
        descArray.add(desc);
        if(kp.toArray().length>0)
            matcher.add(descArray);
        Features2d.drawKeypoints(gray,kp,gray);
        return desc;
    }

    public Mat track(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        Mat gray = getHSVChannels(rgba);

        //calculate key points and descriptors
        Mat desc = new Mat();
        MatOfKeyPoint kp = new MatOfKeyPoint();
        detector.detect(gray,kp);
        extractor.compute(gray,kp,desc);

        //calculate the matches
        // find corresponding matches for the key points and draw them
        if(bCalibrated) {
            ArrayList<MatOfDMatch> matches = new ArrayList<>();
            KeyPoint[] keyPoints = kp.toArray();
            matcher.knnMatch(desc, matches, k);
            // ratio test
            LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
            for (MatOfDMatch dMatches : matches) {
                if (dMatches.toArray()[0].distance / dMatches.toArray()[1].distance < 0.8) {
                    good_matches.add(dMatches.toArray()[0]);
                }
            }
            // get keypoint coordinates of good matches to find homography and remove outliers using ransac
            List<Point> pts1 = new ArrayList<>();
            for(int i = 0; i<good_matches.size(); i++){
                pts1.add(kp.toList().get(good_matches.get(i).queryIdx).pt);
            }

            if(!pts1.isEmpty()) {
                MatOfPoint2f pts = new MatOfPoint2f(pts1.toArray(new Point[pts1.size()]));
                Point center = new Point();
                float[] radii = new float[1];
                Imgproc.minEnclosingCircle(pts, center, radii);
                Imgproc.circle(gray, center, Math.round(radii[0]), new Scalar(255, 0, 0));
                return gray;
            }
        }
        return rgba;
    }

    private List<Mat> hsv(Mat rgba) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(rgba,hsv, Imgproc.COLOR_RGB2HSV);
        List<Mat> channels =new ArrayList<>();
        Core.split(hsv,channels);
        return channels;
    }
}
