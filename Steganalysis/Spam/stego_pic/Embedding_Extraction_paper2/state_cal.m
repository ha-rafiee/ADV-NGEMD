function state=cal_state(n,z,k)
if n==k
    state=(2*z+1)^n;
else if k==0
    state=(2*z-1)^n;
else
    state=2*state_cal(n-1,z,k-1)+(2*z-1)*state_cal(n-1,z,k);
end
end
