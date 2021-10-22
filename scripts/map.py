from slam import PTS
import OpenGL.GL as gl
import cv2
import pangolin
import numpy as np

cap = cv2.VideoCapture("../data/test_countryroad.mp4")
pts = PTS(W=860, H=540, F=200)

def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        
        while cap.isOpened():
            ret, frame = cap.read()
            points = pts.getPoints(frame)
            if len(points) != 0:
                gl.glPointSize(10)
                gl.glColor3f(1.0, 0.0, 0.0)
                # access numpy array directly(without copying data), array should be contiguous.
                pangolin.DrawPoints(points)    
                pangolin.FinishFrame()



if __name__ == '__main__':
    main()


