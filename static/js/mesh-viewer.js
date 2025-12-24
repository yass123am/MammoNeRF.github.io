document.addEventListener("DOMContentLoaded", function () {
  const viewers = [];
  const sceneGroups = [[], [], []];
  window.renderers = [];
  let isUpdating = false;

  // ====== VOXEL TWEAKS ======
  const VOXEL_SIZE = 0.015;     // Smaller = faster + cleaner
  const VOXEL_OPACITY = 0.9;    // Transparency for depth clarity
  const TARGET_SIZE = 1.5;      // Normalized object size
  // =========================

  function init() {
    const containers = [];
    for (let i = 1; i <= 3; i++) {
      containers.push(document.getElementById(`mesh-container-${i}`));
    }

    const modelPaths = [
      './static/mesh/new_v.ply',
      './static/mesh/new_v.ply',
      './static/mesh/new_v.ply',
    ];

    containers.forEach((container, i) => {
      if (!container) return;

      const sceneIndex = Math.floor(i / 1);
      const viewerIndex = i % 1;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xffffff);

      const camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.01,
        100
      );

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      container.appendChild(renderer.domElement);

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      scene.add(new THREE.AmbientLight(0xffffff, 0.8));
      scene.add(new THREE.DirectionalLight(0xffffff, 0.4).position.set(5, 5, 5));
      scene.add(new THREE.DirectionalLight(0xffffff, 0.4).position.set(-5, 5, -5));

      const viewer = {
        scene,
        camera,
        renderer,
        controls,
        container,
        sceneIndex,
        viewerIndex
      };

      viewers.push(viewer);
      sceneGroups[sceneIndex].push(viewer);
      window.renderers.push(renderer);

      const loader = new THREE.PLYLoader();
      loader.load(
        modelPaths[i],
        geometry => {
          // ====== VOXEL GEOMETRY PROCESSING ======
          geometry.computeBoundingBox();

          const center = geometry.boundingBox.getCenter(new THREE.Vector3());
          geometry.translate(-center.x, -center.y, -center.z);

          const size = geometry.boundingBox.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = TARGET_SIZE / maxDim;

          // ====== OPTIONAL PERFORMANCE DECIMATION ======
          // Uncomment if PLY is extremely dense
          /*
          const stride = 2;
          const pos = geometry.attributes.position;
          const col = geometry.attributes.color;

          const newPos = [];
          const newCol = [];

          for (let i = 0; i < pos.count; i += stride) {
            newPos.push(pos.getX(i), pos.getY(i), pos.getZ(i));
            if (col) {
              newCol.push(col.getX(i), col.getY(i), col.getZ(i));
            }
          }

          geometry.setAttribute(
            'position',
            new THREE.Float32BufferAttribute(newPos, 3)
          );

          if (col) {
            geometry.setAttribute(
              'color',
              new THREE.Float32BufferAttribute(newCol, 3)
            );
          }
          */
          // ======================================

          const material = new THREE.PointsMaterial({
            size: VOXEL_SIZE,
            sizeAttenuation: true,
            transparent: true,
            opacity: VOXEL_OPACITY,
            depthWrite: false,
            vertexColors: !!geometry.attributes.color,
            color: geometry.attributes.color ? undefined : 0x3366cc
          });

          const points = new THREE.Points(geometry, material);
          points.scale.setScalar(scale);
          scene.add(points);

          const radius = 2.5;
          camera.position.set(radius, radius * 0.5, radius);
          camera.lookAt(0, 0, 0);
          controls.target.set(0, 0, 0);
          controls.update();
        },
        undefined,
        error => {
          console.error(`Failed to load voxel grid ${modelPaths[i]}`, error);
          const fallback = new THREE.Points(
            new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0)]),
            new THREE.PointsMaterial({ color: 0xff6b6b, size: 0.1 })
          );
          scene.add(fallback);
        }
      );
    });

    setTimeout(setupCameraSync, 1000);
    animate();
  }

  function setupCameraSync() {
    viewers.forEach(viewer => {
      viewer.controls.addEventListener('change', () => {
        if (!isUpdating) {
          syncCamerasInScene(viewer.sceneIndex, viewer.viewerIndex);
        }
      });
    });
  }

  function syncCamerasInScene(sceneIndex, sourceViewerIndex) {
    if (isUpdating) return;
    isUpdating = true;

    const sceneViewers = sceneGroups[sceneIndex];
    const source = sceneViewers[sourceViewerIndex];
    if (!source) return;

    sceneViewers.forEach((viewer, index) => {
      if (index !== sourceViewerIndex) {
        viewer.camera.position.copy(source.camera.position);
        viewer.camera.rotation.copy(source.camera.rotation);
        viewer.controls.target.copy(source.controls.target);
        viewer.controls.update();
      }
    });

    isUpdating = false;
  }

  function animate() {
    requestAnimationFrame(animate);
    viewers.forEach(viewer => {
      if (viewer.container.offsetParent !== null) {
        viewer.controls.update();
        viewer.renderer.render(viewer.scene, viewer.camera);
      }
    });
  }

  window.addEventListener('resize', () => {
    viewers.forEach((viewer, i) => {
      const container = document.getElementById(`mesh-container-${i + 1}`);
      if (container && container.offsetParent !== null) {
        const width = container.clientWidth;
        const height = container.clientHeight;
        viewer.camera.aspect = width / height;
        viewer.camera.updateProjectionMatrix();
        viewer.renderer.setSize(width, height);
      }
    });
  });

  init();
});
