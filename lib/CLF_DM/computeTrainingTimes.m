clear 

%% SEDSII time
Time1 = [0.0020    0.0015    0.0015    0.0015    0.0015    0.0015    0.0015    0.0015    0.0015    0.0015];
Time2 = [  1.9699    1.9621    1.9597    1.9615    1.9617    1.9572    1.9589    1.9609    1.9618    1.9603];
    
TimeGP = Time1;
TimeSEDS = Time1 + Time2;

Time1 = [  0.0022    0.0017    0.0018    0.0017    0.0017    0.0017    0.0017    0.0017    0.0017    0.0017];
Time2 = [  1.0169    1.0125    1.0126    1.0120    1.0115    1.0114    1.0140    1.0112    1.0131    1.0118];

TimeGP = [TimeGP  Time1];
TimeSEDS =  [TimeSEDS  Time1+Time2];

Time1 = [ 0.0021    0.0017    0.0016    0.0016    0.0016    0.0016    0.0017    0.0017    0.0018    0.0017];
Time2 = [ 3.7432    3.8792    4.0925    3.8963    3.8573    3.9366    3.7015    3.7976    3.7484    3.7047];
  
TimeGP = [TimeGP  Time1];
TimeSEDS =  [TimeSEDS  Time1+Time2];

Time1 = [ 0.0038    0.0032    0.0030    0.0032    0.0032    0.0031    0.0031    0.0031    0.0029    0.0029];
Time2 = [ 7.6879    7.6385    7.6211    7.6276    7.6171    7.6209    7.5468    7.6176    7.6135    7.5638 ];

TimeGP = [TimeGP  Time1];
TimeSEDS =  [TimeSEDS  Time1+Time2];

%% R-DS time
TimeRDS = [ 0.0189    0.0106    0.0105    0.0105    0.0105    0.0105    0.0105    0.0105    0.0105    0.0105...
            0.0249    0.0106    0.0106    0.0106    0.0106    0.0106    0.0106    0.0106    0.0106    0.0107...
            0.0193    0.0108    0.0106    0.0107    0.0106    0.0106    0.0106    0.0105    0.0106    0.0106...
            0.0373    0.0308    0.0337    0.0353    0.0410    0.0385    0.0418    0.0434    0.0448    0.0462];


disp(['Mean time GP: ' num2str(mean(TimeGP)) ', std ' num2str(std(TimeGP))]);
disp(['Mean time SEDSII: ' num2str(mean(TimeSEDS)) ', std ' num2str(std(TimeSEDS))]);
disp(['Mean time RDS: ' num2str(mean(TimeRDS)) ', std ' num2str(std(TimeRDS))]);