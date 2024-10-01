function save_vtk(vertices, faces, filename)
    % Open the file for writing
    fid = fopen(filename, 'w');
    
    % Write the header for the VTK file
    fprintf(fid, '# vtk DataFile Version 3.0\n');
    fprintf(fid, '3D surface data\n');
    fprintf(fid, 'ASCII\n');
    fprintf(fid, 'DATASET POLYDATA\n');
    
    % Write the vertices
    num_vertices = size(vertices, 1);
    fprintf(fid, 'POINTS %d float\n', num_vertices);
    for i = 1:num_vertices
        fprintf(fid, '%f %f %f\n', vertices(i, 1), vertices(i, 2), vertices(i, 3));
    end
    
    % Write the faces
    num_faces = size(faces, 1);
    if size(faces, 2) == 3
        % Triangular faces
        fprintf(fid, 'POLYGONS %d %d\n', num_faces, num_faces * 4);
        for i = 1:num_faces
            fprintf(fid, '3 %d %d %d\n', faces(i, 1) - 1, faces(i, 2) - 1, faces(i, 3) - 1);
        end
    elseif size(faces, 2) == 4
        % Quadrilateral faces
        fprintf(fid, 'POLYGONS %d %d\n', num_faces, num_faces * 5);
        for i = 1:num_faces
            fprintf(fid, '4 %d %d %d %d\n', faces(i, 1) - 1, faces(i, 2) - 1, faces(i, 3) - 1, faces(i, 4) - 1);
        end
    end
    
    % Close the file
    fclose(fid);
end