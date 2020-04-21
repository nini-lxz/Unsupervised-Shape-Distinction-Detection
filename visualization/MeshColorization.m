clear all;

txt_name = 'plant_0313.txt'; %'lamp_0128.txt';
pc_saliency = load(txt_name);
pc_pos = pc_saliency(:, 1:3);

% read original mesh      
[V, F] = read_off('plant_0313.off'); %('lamp_0128.off');      
V = V';
F = F';
vertices_num = size(V, 1); 
facets_num = size(F, 1);

% find k=6 nearest points in point cloud for each vertex in V
idx = knnsearch(pc_pos, V, 'K', 6);
% assign color to each vertex (average over nearest k points)
vertices_saliency = zeros(vertices_num, 1);
for i = 1:vertices_num
    vertices_saliency(i, 1) = mean(pc_saliency(idx(i, :), 4));
end

% normalize vertex saliency
vertices_saliency_norm = (vertices_saliency - min(vertices_saliency))./...
                            (max(vertices_saliency)-min(vertices_saliency));



val1 = 0.4; val2 = 0.5; val3 = 0.6;  % adjust the three params to fine-tune the color distribution
vertices_rgbt = zeros(vertices_num, 4);
        
%% for keyshot rendering
for i = 1:vertices_num
    vertices_rgbt(i, 4) = 1;
    if vertices_saliency_norm(i, 1) < val1
        vertices_rgbt(i, 1) = 0;
%                 vertices_rgbt(i, 2) = (1/val1)*vertices_saliency_norm(i, 1);
        vertices_rgbt(i, 2) = (1.0/val1)*vertices_saliency_norm(i, 1)+0.0;
        vertices_rgbt(i, 3) = 0.6;
    elseif vertices_saliency_norm(i, 1) >= val1 && vertices_saliency_norm(i, 1) < val2
        vertices_rgbt(i, 1) = (vertices_saliency_norm(i, 1)-val1)/(val2-val1);
        vertices_rgbt(i, 2) = 1;
        vertices_rgbt(i, 3) = 0.6*(val2-vertices_saliency_norm(i, 1))/(val2-val1);
    elseif vertices_saliency_norm(i, 1) >= val2 && vertices_saliency_norm(i, 1) < val3
        vertices_rgbt(i, 1) = 1;
        vertices_rgbt(i, 2) = (vertices_saliency_norm(i, 1)-val3)/(val2-val3);
        vertices_rgbt(i, 3) = 0;
    else
        vertices_rgbt(i, 1) = (0.2/(val3-1))*vertices_saliency_norm(i, 1)+0.8-(0.2/(val3-1));
        vertices_rgbt(i, 2) = 0;
        vertices_rgbt(i, 3) = 0;
    end
end
        
%%
% convert to (0, 255)
vertices_rgbt = round(vertices_rgbt.*255);

% save to a .off file
output_path = 'plant_0313_colorize.off'; %'lamp_0128_colorize.off';
fid = fopen(output_path, 'w');
fprintf(fid, 'COFF\n');
fprintf(fid, '%d %d %d\n', vertices_num, facets_num, 0);
for i = 1:vertices_num
    fprintf(fid, '%f %f %f %d %d %d %d\n', V(i, :), vertices_rgbt(i, :));
end
for i = 1:facets_num
    fprintf(fid, '%d %d %d %d\n', 3, F(i, 1)-1, F(i, 2)-1, F(i, 3)-1);
end
fclose(fid);







