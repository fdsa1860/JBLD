function d = dist_JBLD(X,Y)
d = logdetM((X+Y)/2)-0.5*logdetM(X)-0.5*logdetM(Y);
end