function show_skel_JHMDB(data)
%   data[2JXT]: 2D skeletion data of J joints and T frames

J = [1,1,1,1,2,2,4,5,6,7,8,9,10,11
    2,3,4,5,6,7,8,9,10,11,12,13,14,15];


x = data(1:2:end,:);
y = data(2:2:end,:);

T = size(data,2);

% clf;
% figure;

for t=1:T
    plot(x(:,t),y(:,t),'o');
    set(gca,'Ydir','reverse');
%     axis([min(min(x)) max(max(x)) min(min(y)) max(max(y))]);
    axis([-1 1 -1 1]);
    for j=1:14
        c1=J(1,j);
        c2=J(2,j);
        line([x(c1,t) x(c2,t)], [y(c1,t) y(c2,t)]);
    end    
    title([num2str(t) '/' num2str(T)]);
%     axis tight;
%     axis off;
%     view(-30,30);
    drawnow;
%     pause(0.1);
    t
    pause;
    35;
end

end

