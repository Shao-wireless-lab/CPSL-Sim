function epoch = UTC2epoch(y) 
% UTC2epoch('22-May-2017 08:00:00') 
epoch = seconds(datetime(y)-datetime('01-Jan-1970 00:00:00')); 
end