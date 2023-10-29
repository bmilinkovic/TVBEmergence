
di_directory = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData"

% Get a list of all files in the directory
files = dir(di_directory);

% Loop through each file and check if it contains "dynamical_dependence" in the name
for i = 1:length(files)
    file_name = files(i).name;
    if contains(file_name, "dynamical_dependence")
        % Load the file
        file_path = fullfile(di_directory, file_name);
        data = load(file_path);
        % Do something with the data here
    end
end

