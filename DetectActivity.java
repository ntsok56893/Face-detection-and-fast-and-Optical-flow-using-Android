package com.example.a05.imgfd;

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.Video;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

public class DetectActivity extends AppCompatActivity implements
        CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {

    private CameraBridgeViewBase cameraView;
    private CascadeClassifier classifier;
    private FeatureDetector fast;
    private Mat mGray;
    private Mat mRgba;
    private Mat premRgba;
    private Mat preFrame;
    private int mAbsoluteFaceSize = 0;
    private boolean isFrontCamera;
    private MatOfKeyPoint points;
    private MatOfPoint2f curpoints;
    private MatOfPoint2f prepoints;
    private List<KeyPoint> listOfKeypoints;
    private List<KeyPoint> listOfBestKeypoints;
    private LinkedList<Point> pointList;

    private MatOfByte status;
    private MatOfFloat err;

    private int frameCount;
    private int frameThreshold;
    private int y;

    static {
        System.loadLibrary("opencv_java3");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        initWindowSettings();
        setContentView(R.layout.activity_detect);
        cameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this); // 設置相機監聽
        initClassifier();
        fast = FeatureDetector.create(FeatureDetector.FAST);
        cameraView.enableView();
        cameraView.enableFpsMeter();
        Button switchCamera = (Button) findViewById(R.id.switch_camera);
        switchCamera.setOnClickListener(this); // 切換相機鏡頭，默認後置相機
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.switch_camera:
                cameraView.disableView();
                if (isFrontCamera) {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                    isFrontCamera = false;
                } else {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                    isFrontCamera = true;
                }
                cameraView.enableView();
                break;
            default:
        }
    }

    // 初始化窗口設定
    private void initWindowSettings() {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
    }

    // 初始化人臉級聯分類器
    private void initClassifier() {
        try {
            InputStream is = getResources()
                    .openRawResource(R.raw.haarcascade_frontalface_alt2);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        preFrame = new Mat();
        premRgba = new Mat();

        points = new MatOfKeyPoint();
        curpoints = new MatOfPoint2f();
        prepoints = new MatOfPoint2f();
        status = new MatOfByte();
        err = new MatOfFloat();


        pointList = new LinkedList<>();

        frameCount = 0;
        frameThreshold = 15;
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        // 翻轉矩陣，用來校正前後鏡頭
        if (isFrontCamera) {
            Core.flip(mRgba, mRgba, 1);
            Core.flip(mGray, mGray, 1);
        }
        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        MatOfRect faces = new MatOfRect();

        if (classifier != null)
            classifier.detectMultiScale(mGray, faces, 1.1, 3, 0,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255);
        Scalar faceDectColor = new Scalar(255, 0, 0, 255);

        for (Rect faceRect : facesArray) {

            mRgba.submat(faceRect).copyTo(preFrame);

            fast.detect(preFrame, points);


            listOfKeypoints = points.toList();

            Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
                @Override
                public int compare(KeyPoint kp1, KeyPoint kp2) {
                    // Sort them in descending order, so the best response KPs will come first
                    return (int) (kp2.response - kp1.response);
                }
            });

            if(listOfKeypoints.size() < 500){
                listOfBestKeypoints = listOfKeypoints.subList(0, listOfKeypoints.size());
            }
            else{
                listOfBestKeypoints = listOfKeypoints.subList(0, 500);
            }

            points.fromList(listOfBestKeypoints);

            pointList.clear();

            for (int i = 0; i < listOfBestKeypoints.size(); i++) {
                listOfBestKeypoints.get(i).pt.setX(listOfBestKeypoints.get(i).pt.getX() + faceRect.tl().getX());
                listOfBestKeypoints.get(i).pt.setY(listOfBestKeypoints.get(i).pt.getY() + faceRect.tl().getY());
            }

            for (int i = 0; i < listOfBestKeypoints.size(); i++) {
                pointList.add(listOfBestKeypoints.get(i).pt);
            }

            curpoints.fromList(pointList);

            if(frameCount >= frameThreshold){
                mRgba.copyTo(premRgba);

                prepoints.fromList(curpoints.toList());
                //Imgproc.putText(mRgba, String.valueOf(frameCount), new Point(200, 200), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(255, 0, 0));
                //Imgproc.circle(mRgba, prepoints.toList().get(5), 5, faceDectColor, 3);
                frameCount = 0;
            }

            Imgproc.rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3);

            if(!faceRect.empty()){
                break;
            }
        }

        if(!premRgba.empty()){


            Video.calcOpticalFlowPyrLK(premRgba, mRgba, prepoints, curpoints, status, err);

            y = status.toList().size() - 1;

            for (int x = 0; x < y; x++) {
                if (status.toList().get(x) == 1) {
                    Imgproc.circle(mRgba, curpoints.toList().get(x), 5, faceDectColor, 3);

                    Imgproc.line(mRgba, prepoints.toList().get(x), curpoints.toList().get(x), faceDectColor, 3);
                }
            }
        }
        frameCount++;
        return mRgba;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraView.disableView();
    }
}
