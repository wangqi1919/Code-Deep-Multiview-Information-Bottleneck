clc;
clear;
dir = 'synthetic_data/';
mkdir(dir);
for i = 1: 5
rng(i);
data_variance = 1;
feature_dim = 40;
extra_feature_dim = 20;
sample_size = 4000;
class_shift = 1;
extract_feature_level =1;
Noise_level = 1;
% first create two classes as the hidden variable
latent_c1 = mvnrnd(class_shift/2 * ones(1, feature_dim), data_variance * eye(feature_dim), sample_size/2);
latent_c2 = mvnrnd(-class_shift/2 * ones(1, feature_dim), data_variance * eye(feature_dim), sample_size/2);
latent = [latent_c1; latent_c2];


extract_feature1_c1 = mvnrnd(-extract_feature_level * ones(1, extra_feature_dim), eye(extra_feature_dim), round(sample_size/3));
extract_feature1_c2 = mvnrnd(extract_feature_level * ones(1, extra_feature_dim), eye(extra_feature_dim), round(sample_size/3));
extract_feature1_c3 = mvnrnd(-extract_feature_level * ones(1, extra_feature_dim), eye(extra_feature_dim), sample_size - round(sample_size/3) - round(sample_size/3));
extra_feature1 = [extract_feature1_c1; extract_feature1_c2; extract_feature1_c3];

%extra_feature1 = mvnrnd(-1 * ones(1, extra_feature_dim), extract_feature_level * eye(extra_feature_dim), sample_size);
extract_feature2_c1 = mvnrnd(-extract_feature_level * ones(1, extra_feature_dim), eye(extra_feature_dim), round(sample_size/6));
extract_feature2_c2 = mvnrnd(extract_feature_level * ones(1, extra_feature_dim), eye(extra_feature_dim),  sample_size - round(sample_size/6) - round(sample_size/6));
extract_feature2_c3 = mvnrnd(-extract_feature_level * ones(1, extra_feature_dim), eye(extra_feature_dim), round(sample_size/6));
extra_feature2 = [extract_feature2_c1; extract_feature2_c2; extract_feature2_c3];

%extra_feature2 = mvnrnd(ones(1, extra_feature_dim), extract_feature_level * eye(extra_feature_dim), sample_size);


x1 = [latent, extra_feature1];
x1 = tanh((tanh(x1)) + 0.1);
Noise_level1 = Noise_level * max(abs(max(max(x1))), abs(min(min(x1))));
Noise1 = mvnrnd(zeros(1, feature_dim+extra_feature_dim), Noise_level1 * eye(feature_dim+extra_feature_dim), sample_size);
x1 = x1 + Noise1;
x2 = [latent, extra_feature2];
x2 = sigmf(x2, [1, 0]) - 0.5;
Noise_level2 = Noise_level * max(abs(max(max(x2))), abs(min(min(x2))));
Noise2 = mvnrnd(zeros(1, feature_dim+extra_feature_dim), Noise_level2 * eye(feature_dim+extra_feature_dim), sample_size);
x2 = x2 + Noise2;

x1 = zscore(x1);
x2 = zscore(x2);

% plot out
% [coeff, socre] = pca(latent);
% reduced = socre(:, 1:2);
% figure;
% scatter(reduced(1: sample_size/2, 1), reduced(1: sample_size/2,2), 'filled');
% hold on
% scatter(reduced(sample_size/2+1: end, 1), reduced(sample_size/2+1: end,2), 'filled');

[coeff, socre] = pca(extra_feature1);
reduced = socre(:, 1:2);
figure;
scatter(reduced(1: sample_size/2, 1), reduced(1: sample_size/2,2), 'filled');
hold on
scatter(reduced(sample_size/2+1: end, 1), reduced(sample_size/2+1: end,2), 'filled');



[coeff, socre] = pca(x1);
reduced = socre(:, 1:2);
figure;
scatter(reduced(1: sample_size/2, 1), reduced(1: sample_size/2,2), 'filled');
hold on
scatter(reduced(sample_size/2+1: end, 1), reduced(sample_size/2+1: end,2), 'filled');

[coeff, socre] = pca(x2);
reduced = socre(:, 1:2);
figure;
scatter(reduced(1: sample_size/2, 1), reduced(1: sample_size/2,2), 'filled');
hold on
scatter(reduced(sample_size/2+1: end, 1), reduced(sample_size/2+1: end,2), 'filled');

y = [zeros(sample_size/2, 1); ones(sample_size/2, 1)];

% split data to training and testing
train_size = round(min(nnz(y==0), nnz(y == 1)) * 0.6);
val_size = round(min(nnz(y==0), nnz(y == 1)) * 0.2);

idx_pos = find(y == 0);
idx_pos = idx_pos(randperm(length(idx_pos)));

idx_neg = find(y == 1);
idx_neg = idx_neg(randperm(length(idx_neg)));

tr_idx = [idx_pos(1: train_size); idx_neg(1: train_size)];
val_idx = [idx_pos(train_size + 1: train_size + val_size); idx_neg(train_size + 1: train_size + val_size)];
te_idx = [idx_pos(train_size + val_size + 1 : end); idx_neg(train_size + val_size + 1 : end)];
X1_train = x1(tr_idx, :);
X2_train = x2(tr_idx, :);
ytrain = y(tr_idx, :);
X1_val = x1(val_idx, :);
X2_val = x2(val_idx, :);
yval= y(val_idx, :);
X1_test = x1(te_idx, :);
X2_test = x2(te_idx, :);
ytest= y(te_idx, :);
save(strcat(dir, 'iter', int2str(i)), 'X1_train', 'X1_test', 'X1_val', 'X2_train', 'X2_test', 'X2_val','ytrain', 'ytest', 'yval');
end

