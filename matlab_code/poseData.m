%% notes for alex - loading data in matlab

load('/Users/ugne/Dropbox/ratDNN/trainTestSplit.mat')

skeleton.joints = [{'HeadF'    }
    {'HeadB'    }
    {'HeadL'    }
    {'SpineF'   }
    {'SpineM'   }
    {'SpineL'   }
    {'Offset1'  }
    {'Offset2'  }
    {'HipL'     }
    {'HipR'     }
    {'ElbowL'   }
    {'ArmL'     }
    {'ShoulderL'}
    {'ShoulderR'}
    {'ElbowR'   }
    {'ArmR'     }
    {'KneeR'    }
    {'KneeL'    }
    {'ShinL'    }
    {'ShinR'    }];

skeleton.joints_idx =[1     2
     2     3
     1     3
     2     4
     1     4
     3     4
     4     5
     5     6
     4     7
     7     8
     5     8
     5     7
     6     8
     6     9
     6    10
    11    12
     4    13
     4    14
    11    13
    12    13
    14    15
    14    16
    15    16
     9    18
    10    17
    18    19
    17    20];
    

skeleton.color = zeros(27,3);
skeleton.mcolor = zeros(20,3);

i = 100; 
markers = squeeze(split1True(i,:,:))';
mshuffled = squeeze(split1(i,:,:))';



figure(1);
subplot(2,1,1);
for i = 1:20 
    scatter3(markers(:,1),markers(:,2),markers(:,3),[],1:20,'filled'); hold on; 
end
for i = 1:20
    for j = 1:20
        idx = [i j];
        plot3(markers(idx,1),markers(idx,2),markers(idx,3));
    end
end

for i = 1:27
    idx = skeleton.joints_idx(i,:);
    plot3(markers(idx,1),markers(idx,2),markers(idx,3));
end
axis equal 


subplot(2,1,2);
for i = 1:20 
    scatter3(mshuffled(:,1),mshuffled(:,2),mshuffled(:,3),[],1:20,'filled'); hold on; 
end
for i = 1:27
    idx = skeleton.joints_idx(i,:);
    plot3(mshuffled(idx,1),mshuffled(idx,2),mshuffled(idx,3));
end
axis equal 



samples = [1 2 3 4];
for i = 1:4
    m = squeeze(split1True(samples(i),:,:))';
    subplot(2,2,i);
    scatter3(m(:,1),m(:,2),m(:,3),[],1:20,'filled'); hold on; 
    
    for j = 1:27
    idx = skeleton.joints_idx(j,:);
    plot3(m(idx,1),m(idx,2),m(idx,3));
    end
    for j = 1:20
    text(m(j,1),m(j,2),m(j,3),num2str(j));
    end
end

samples = [1 2 3 4];
for i = 1:4
    m = squeeze(split1(samples(i),:,:))';
    subplot(2,2,i);
    scatter3(m(:,1),m(:,2),m(:,3),[],1:20,'filled'); hold on; 
    for j = 1:20
    text(m(j,1),m(j,2),m(j,3),num2str(j));
    end
    %for j = 1:27
    %idx = skeleton.joints_idx(j,:);
    %plot3(markers(idx,1),markers(idx,2),markers(idx,3));
    %end
end

h{1} = Keypoint3DAnimator(split1True, skeleton, 'Position', [0 0  1 1]);

axis equal;




markers = squeeze(split1True(1,:,:))'
rp = randperm(17)+3;
markers(4:end,:) = markers(rp,:);


p = pdist2(markers,markers);




