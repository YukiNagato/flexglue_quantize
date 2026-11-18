"""Geometry metrics with reference pinhole/OpenCV-distorted cameras."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import cv2
import numpy as np


def quaternion_rotation(q):
    w,x,y,z=np.asarray(q,dtype=np.float64)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def cross_matrix(t):
    x,y,z=t
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=np.float64)


def relative_pose(image0,image1):
    return np.linalg.inv(np.asarray(image1['camera_to_world']))@np.asarray(image0['camera_to_world'])


def validate_camera(image):
    model=image.get('camera_model','PINHOLE')
    if model not in ('PINHOLE','OPENCV'):
        raise ValueError(f'Unsupported reference camera model: {model}')
    d=image.get('distortion',{})
    if model=='OPENCV' and any(abs(float(d.get(k,0)))>0 for k in ['k4','k5','k6']):
        raise ValueError('Rational distortion requires a FULL_OPENCV reference model')


def undistort_points(points,image,normalized=False):
    """Ideal pixels for Sampson error, or normalized rays for pose estimation.

    Model inputs/predicted keypoints remain in the original image coordinates.
    Reference distortion is used only by the CPU geometry scoring step.
    """
    validate_camera(image)
    points=np.asarray(points,dtype=np.float64).reshape(-1,2)
    if not len(points):return points
    K=np.asarray(image['K'],dtype=np.float64)
    d=image.get('distortion',{})
    coeff=np.array([d.get(k,0.) for k in ['k1','k2','p1','p2','k3']],dtype=np.float64)
    if image.get('camera_model','PINHOLE')=='PINHOLE' or not np.any(coeff):
        if not normalized:return points
        return cv2.undistortPoints(points.reshape(-1,1,2),K,None)[:,0]
    criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-12)
    if hasattr(cv2,'undistortPointsIter'):
        return cv2.undistortPointsIter(points.reshape(-1,1,2),K,coeff,None,
            None if normalized else K,criteria)[:,0]
    # OpenCV 5 folds the iterative overload into undistortPoints.
    return cv2.undistortPoints(points.reshape(-1,1,2),K,coeff,
        R=None,P=None if normalized else K,criteria=criteria)[:,0]


def sampson_distance_px(points0,points1,image0,image1):
    T=relative_pose(image0,image1)
    E=cross_matrix(T[:3,3])@T[:3,:3]
    F=np.linalg.inv(np.asarray(image1['K'])).T@E@np.linalg.inv(np.asarray(image0['K']))
    a=np.c_[undistort_points(points0,image0),np.ones(len(points0))]
    b=np.c_[undistort_points(points1,image1),np.ones(len(points1))]
    Fa=a@F.T;Ftb=b@F
    residual=np.sum(b*Fa,axis=1)
    denominator=(Fa[:,:2]**2).sum(1)+(Ftb[:,:2]**2).sum(1)
    return np.abs(residual)/np.sqrt(np.maximum(denominator,1e-30))


def pose_errors(points0,points1,image0,image1,seed=0,ransac_px=1.):
    if len(points0)<5:return dict(success=False,reason='fewer_than_5_matches')
    K0,K1=[np.asarray(m['K'],dtype=np.float64) for m in (image0,image1)]
    a=undistort_points(points0,image0,normalized=True)
    b=undistort_points(points1,image1,normalized=True)
    focal=np.mean([K0[0,0],K0[1,1],K1[0,0],K1[1,1]])
    cv2.setRNGSeed(seed)
    E,mask=cv2.findEssentialMat(a,b,np.eye(3),method=cv2.RANSAC,prob=.999,
                               threshold=ransac_px/focal,maxIters=1000)
    if E is None or mask is None:return dict(success=False,reason='essential_estimation_failed')
    T=relative_pose(image0,image1);Rgt=T[:3,:3];tgt=T[:3,3]
    if np.linalg.norm(tgt)<1e-8:return dict(success=False,reason='zero_reference_baseline')
    best=None
    for essential in np.asarray(E).reshape(-1,3,3):
        try:inliers,R,t,_=cv2.recoverPose(essential,a,b,np.eye(3),mask=mask.copy())
        except cv2.error:continue
        r=float(np.degrees(np.arccos(np.clip((np.trace(R@Rgt.T)-1)/2,-1,1))))
        cosine=abs(float(np.dot(t[:,0],tgt)/(np.linalg.norm(t)*np.linalg.norm(tgt))))
        tr=float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
        row=dict(success=True,rotation_deg=r,translation_deg=tr,error_deg=max(r,tr),inliers=int(inliers))
        # Select pose by cheirality inlier count; ground truth is used only to score it.
        if best is None or row['inliers']>best['inliers']:best=row
    return best or dict(success=False,reason='pose_recovery_failed')


def pose_auc(errors,threshold):
    """Area under the empirical recall curve, divided by threshold."""
    errors=np.sort(np.asarray(errors,dtype=float))
    if not len(errors):return None
    good=errors[errors<=threshold]
    x=np.r_[0.,good,threshold]
    y=np.r_[0.,np.arange(1,len(good)+1)/len(errors),len(good)/len(errors)]
    return float(np.trapezoid(y,x)/threshold)
