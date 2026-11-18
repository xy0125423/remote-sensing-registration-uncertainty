"""原生格网响应边界计算；数学函数取自最终技术封版。"""
from __future__ import annotations
import numpy as np
import shapely

SUPPORTS = ['center', 'bilinear', 'circle15', 'square_equal_area']
CS = [0, 2, 5, 10, 20]
LS = [0., .25, .5, .75, 1.]
PAD = dict(zip(SUPPORTS, [0., 20., 15., 15*np.sqrt(np.pi)/2]))
yc, xc = np.mgrid[:11, :11]
XC, YC = xc*20+10, yc*20+10
yy, xx = np.mgrid[-1:2, -1:2]
DESIGN = np.c_[np.ones(9), xx.ravel()*20, yy.ravel()*20, .5*(xx.ravel()*20)**2, xx.ravel()*yy.ravel()*400, .5*(yy.ravel()*20)**2]
INV = np.linalg.pinv(DESIGN)


def weights(x: np.ndarray, y: np.ndarray, support: str) -> np.ndarray:
    """Vectorize the exact frozen 256-sided polygon, with no resampling."""
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    n = len(x); w = np.zeros((n, 121))
    if support == 'center':
        cc, rr = (x//20).astype(int), (y//20).astype(int)
        assert ((cc>=0)&(cc<11)&(rr>=0)&(rr<11)).all()
        w[np.arange(n), rr*11+cc] = 1
    elif support == 'bilinear':
        cc, rr = x/20-.5, y/20-.5
        c0, r0 = np.floor(cc).astype(int), np.floor(rr).astype(int)
        dc, dr = cc-c0, rr-r0
        for i,j,v in [(0,0,(1-dc)*(1-dr)),(1,0,dc*(1-dr)),(0,1,(1-dc)*dr),(1,1,dc*dr)]:
            assert ((c0+i>=0)&(c0+i<11)&(r0+j>=0)&(r0+j<11)).all()
            w[np.arange(n),(r0+j)*11+c0+i] = v
    else:
        h = 15 if support == 'circle15' else PAD[support]
        shape = shapely.buffer(shapely.points(x,y),15,quad_segs=64) if support=='circle15' else shapely.box(x-h,y-h,x+h,y+h)
        c0, r0 = np.floor((x-h)/20).astype(int), np.floor((y-h)/20).astype(int)
        for j in range(3):
            for i in range(3):
                cc, rr = c0+i, r0+j
                area = shapely.area(shapely.intersection(shape,shapely.box(cc*20,rr*20,(cc+1)*20,(rr+1)*20)))
                good=(cc>=0)&(cc<11)&(rr>=0)&(rr<11)
                assert (area[~good]<1e-12).all()
                w[np.flatnonzero(good),rr[good]*11+cc[good]]=area[good]
        assert np.max(abs(w.sum(1)/shapely.area(shape)-1))<1e-9
        w/=w.sum(1)[:,None]
    assert np.isfinite(w).all() and w.min()>=0 and np.max(abs(w.sum(1)-1))<1e-10
    return w

def state(lo: float, hi: float, valid: bool=True) -> str:
    """Classify an absolute index interval using the fixed zero threshold."""
    if not valid or not np.isfinite([lo,hi]).all() or lo>hi: return '无效'
    return '保证为正' if lo>0 else '保证为负' if hi<0 else '无法保证'

def evaluate(A: np.ndarray, B: np.ndarray, px: float, py: float, rv: np.ndarray, support: str) -> list[dict]:
    """Rebuild reachable pixels and both remainder budgets for one relative vector."""
    ca,cb=INV@A[4:7,4:7].ravel(),INV@B[4:7,4:7].ravel()
    ha=np.array([[ca[3],ca[4]],[ca[4],ca[5]]]);hb=np.array([[cb[3],cb[4]],[cb[4],cb[5]]])
    off=np.array([px-110,py-110]);ga,gb=ca[1:3]+ha@off,cb[1:3]+hb@off
    w=weights([px],[py],support)[0].reshape(11,11);a0,b0=float(np.sum(w*A)),float(np.sum(w*B))
    pa=a0+ga[0]*(XC-px)+ga[1]*(YC-py);pb=b0+gb[0]*(XC-px)+gb[1]*(YC-py)
    moment=np.array([np.sum(w*XC)-px,np.sum(w*YC)-py]);phase=0 if support=='bilinear' else 10*np.sqrt(2)
    e0a,e0b=abs(np.sum(w*(A-pa))),abs(np.sum(w*(B-pb)))
    segments=[];uas=[];ubs=[];pos=0
    for C in CS:
        for lam in LS:
            angles=np.array([0.]) if C==0 else np.arange(16)*2*np.pi/16
            cv=np.c_[C*np.cos(angles),C*np.sin(angles)]
            ua,ub=cv-lam*rv,cv+(1-lam)*rv
            segments.append((C,lam,slice(pos,pos+len(cv)),cv));pos+=len(cv);uas.extend(ua);ubs.extend(ub)
    ua,ub=np.asarray(uas),np.asarray(ubs)
    R=20+np.linalg.norm(rv)+PAD[support]
    full=px-R>=0 and py-R>=0 and px+R<220 and py+R<220
    # Full support is checked for each C below. Invalid large-C conditions are not clipped.
    aa=np.full(len(ua),np.nan);bb=aa.copy()
    goodA=(px+ua[:,0]-PAD[support]>=0)&(py+ua[:,1]-PAD[support]>=0)&(px+ua[:,0]+PAD[support]<220)&(py+ua[:,1]+PAD[support]<220)
    goodB=(px+ub[:,0]-PAD[support]>=0)&(py+ub[:,1]-PAD[support]>=0)&(px+ub[:,0]+PAD[support]<220)&(py+ub[:,1]+PAD[support]<220)
    if goodA.any():aa[goodA]=weights(px+ua[goodA,0],py+ua[goodA,1],support)@A.ravel()
    if goodB.any():bb[goodB]=weights(px+ub[goodB,0],py+ub[goodB,1],support)@B.ravel()
    rows=[]
    for C,lam,sl,cv in segments:
        R=C+np.linalg.norm(rv)+PAD[support]
        low=np.floor((np.array([px,py])-R)/20).astype(int);high=np.floor((np.array([px,py])+R)/20).astype(int)+1
        complete=(low>=0).all() and (high<=11).all()
        valid=bool(complete and np.isfinite(A).all() and np.isfinite(B).all() and goodA[sl].all() and goodB[sl].all())
        for index in ['dNBR','RBR']:
            z0=a0-b0 if index=='dNBR' else (a0-b0)/(a0+1.001)
            row={'support':support,'index':index,'C_m':C,'lambda':lam,'D0':z0,'valid':valid,'invalid_reason':'' if valid else '不完整支持或无效原生窗口','r_east_m':rv[0],'r_north_m':-rv[1]}
            if not valid:
                row.update({'old_state':'无效','new_state':'无效'});rows.append(row);continue
            cell=np.s_[low[1]:high[1],low[0]:high[0]]
            amin,amax=float(A[cell].min()),float(A[cell].max());bmin,bmax=float(B[cell].min()),float(B[cell].max())
            ea=float(abs(A[cell]-pa[cell]).max())+phase*np.linalg.norm(ga)
            eb=float(abs(B[cell]-pb[cell]).max())+phase*np.linalg.norm(gb)
            oa=ea+e0a+abs(ga@moment);ob=eb+e0b+abs(gb@moment)
            if index=='dNBR':
                gc=ga-gb;intercept=-float((lam*ga+(1-lam)*gb)@rv)
                direct=aa[sl]-bb[sl]-z0;lin=ua[sl]@ga-ub[sl]@gb
                nativeLo,nativeHi=amin-bmax-z0,amax-bmin-z0
                oldBudget,newBudget=oa+ob,ea+eb
            else:
                valid=amin+1.001>0 and a0+1.001>0
                if not valid:row.update({'valid':False,'invalid_reason':'RBR非正分母域','old_state':'无效','new_state':'无效'});rows.append(row);continue
                den=a0+1.001;fa=(b0+1.001)/den**2;fb=-1/den
                gc=fa*ga+fb*gb;intercept=float((-lam*fa*ga+(1-lam)*fb*gb)@rv)
                direct=(aa[sl]-bb[sl])/(aa[sl]+1.001)-z0;lin=fa*(ua[sl]@ga)+fb*(ub[sl]@gb)
                nativeLo,nativeHi=(amin-bmax)/(amin+1.001)-z0,(amax-bmin)/(amax+1.001)-z0
                da,db=max(abs(amin-a0),abs(amax-a0)),max(abs(bmin-b0),abs(bmax-b0))
                rem=(bmax+1.001)*da*da/(amin+1.001)**3+da*db/(amin+1.001)**2
                oldBudget,newBudget=abs(fa)*oa+abs(fb)*ob+rem,abs(fa)*ea+abs(fb)*eb+rem
            lo,hi=intercept-C*np.linalg.norm(gc),intercept+C*np.linalg.norm(gc)
            oldLo,oldHi=max(lo-oldBudget,nativeLo),min(hi+oldBudget,nativeHi)
            newLo,newHi=max(lo-newBudget,nativeLo),min(hi+newBudget,nativeHi)
            assert newLo<=newHi and oldLo<=oldHi
            assert direct.min()>=newLo-1e-9 and direct.max()<=newHi+1e-9
            imn,imx=int(np.argmin(direct)),int(np.argmax(direct))
            row.update({'linear_center':intercept,'linear_lower':lo,'linear_upper':hi,'linear_width':hi-lo,
                        'native_lower':nativeLo,'native_upper':nativeHi,'native_width':nativeHi-nativeLo,
                        'old_budget':oldBudget,'new_budget':newBudget,'old_expanded_lower':lo-oldBudget,'old_expanded_upper':hi+oldBudget,
                        'new_expanded_lower':lo-newBudget,'new_expanded_upper':hi+newBudget,
                        'old_lower':oldLo,'old_upper':oldHi,'old_width':oldHi-oldLo,'new_lower':newLo,'new_upper':newHi,'new_width':newHi-newLo,
                        'absolute_lower':z0+newLo,'absolute_upper':z0+newHi,'response_center':(newLo+newHi)/2,
                        'grid_min':float(direct.min()),'grid_max':float(direct.max()),'first_MAE':float(abs(direct-lin).mean()),
                        'old_state':state(z0+oldLo,z0+oldHi),'new_state':state(z0+newLo,z0+newHi),
                        'numerical_flip':bool(z0+direct.min()<0<z0+direct.max()),'min_cx':cv[imn,0],'min_cy':cv[imn,1],'max_cx':cv[imx,0],'max_cy':cv[imx,1],
                        'reachable_col_min':int(low[0]),'reachable_col_end':int(high[0]),'reachable_row_min':int(low[1]),'reachable_row_end':int(high[1]),
                        'lower_active':'像元' if nativeLo>=lo-newBudget else '简化余项','upper_active':'像元' if nativeHi<=hi+newBudget else '简化余项'})
            rows.append(row)
    return rows
