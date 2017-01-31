function show_joints_JHMDB(data, index)
%   data[2JXT]: 2D skeletion data of J joints and T frames

% J = [1,1,1,1,2,2,4,5,6,7,8,9,10,11
%     2,3,4,5,6,7,8,9,10,11,12,13,14,15];


x = data(1:2:end,:);
y = data(2:2:end,:);
% z = data(3:3:end,:);

T = size(data,2);

% clf;

% figure;
% hold on;

for t=1:T
    plot(x(index,t),y(index,t),'o');
    if t > 1
        line(x(index,1:t)', y(index,1:t)');
    end
    set(gca,'Ydir','reverse');
%     set(gca,'DataAspectRatio',[1 1]);
%     axis([min(min(x)) max(max(x)) min(min(y)) max(max(y))]);
    axis([-1 1 -1 1]);
    title(['Display joints on Frame ' num2str(t) '/' num2str(T)]);
%     axis tight;
%     axis off;
%     view(-30,30);
    drawnow;
%     pause(0.1);
    t
    pause;
    35;
end
% hold off;

end

