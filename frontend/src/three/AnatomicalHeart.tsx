// Optional real heart mesh. Drop any heart model at public/models/heart.glb and the app
// offers an "Anatomical" mode. The mesh is auto-centred and auto-scaled to the procedural
// heart's size, then fine-tuned with the Align panel (saved per browser, exportable as
// public/models/heart.json so the whole team gets the same alignment).
//
// The coronary tubes stay procedural so their colours remain tied 1:1 to the model outputs.
// The mesh is rendered as a translucent hologram so the vessels are visible even where the
// mesh's own surface sits slightly outside them.

import { useMemo } from 'react';
import { useGLTF } from '@react-three/drei';
import {
  AdditiveBlending, BackSide, Box3, Color, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, ShaderMaterial, Vector3,
  type Object3D,
} from 'three';
import { PALETTE, SHAPE } from './anatomy';

import { DEFAULT_ALIGN, MODEL_URL, type Align } from './modelConfig';
export { DEFAULT_ALIGN, MODEL_URL, type Align };

const deg = (d: number) => (d * Math.PI) / 180;
// Target box: roughly the procedural myocardium plus great vessels.
const TARGET_HEIGHT = 3.0;
const TARGET_CENTER = new Vector3(0, SHAPE.cy + 0.35, 0);

export default function AnatomicalHeart({ align, url = MODEL_URL }: { align: Align; url?: string }) {
  const gltf = useGLTF(url);

  const { body, wire, rim, fit } = useMemo(() => {
    const tissue = new MeshPhysicalMaterial({
      color: new Color(PALETTE.tissue), roughness: 0.5, metalness: 0, clearcoat: 0.45,
      transparent: true, opacity: 0.62, depthWrite: false,
    });
    const wireMat = new MeshBasicMaterial({ color: '#ffd0c0', wireframe: true, transparent: true, opacity: 0.03, depthWrite: false });
    const rimMat = new ShaderMaterial({
      uniforms: { color: { value: new Color(PALETTE.rim) } },
      vertexShader: `varying vec3 vN; varying vec3 vV;
        void main(){ vec4 mv = modelViewMatrix*vec4(position,1.0); vec3 n = normalMatrix*normal; vN = length(n) > 1e-5 ? normalize(n) : vec3(0.0,0.0,1.0); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `uniform vec3 color; varying vec3 vN; varying vec3 vV;
        void main(){ float f = pow(clamp(1.0-abs(dot(vN,vV)),0.0,1.0), 2.6); gl_FragColor = vec4(color*f*1.4, f*0.9); }`,
      transparent: true, blending: AdditiveBlending, depthWrite: false, side: BackSide,
    });
    const restyle = (root: Object3D, m: MeshPhysicalMaterial | MeshBasicMaterial | ShaderMaterial) => {
      root.traverse((o) => {
        if ((o as Mesh).isMesh) {
          const mesh = o as Mesh;
          // Some exported models ship without normals; lighting and the rim shader need them.
          if (!mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
          mesh.material = m;
          mesh.raycast = () => {}; // clicks go to the vessels, not the shell
        }
      });
      return root;
    };
    const b = restyle(gltf.scene.clone(true), tissue);
    const w = restyle(gltf.scene.clone(true), wireMat);
    const r = restyle(gltf.scene.clone(true), rimMat);

    // Auto-fit: centre and scale the mesh's bounding box onto the procedural heart.
    const box = new Box3().setFromObject(gltf.scene);
    const size = box.getSize(new Vector3());
    const center = box.getCenter(new Vector3());
    const s = TARGET_HEIGHT / Math.max(size.x, size.y, size.z, 1e-6);
    return { body: b, wire: w, rim: r, fit: { s, center } };
  }, [gltf]);

  const k = fit.s * align.scale;
  return (
    <group
      position={[TARGET_CENTER.x + align.x, TARGET_CENTER.y + align.y, TARGET_CENTER.z + align.z]}
      rotation={[deg(align.rx), deg(align.ry), deg(align.rz)]}
    >
      <group scale={k} position={fit.center.clone().multiplyScalar(-k)}>
        <primitive object={body} />
        <primitive object={wire} />
      </group>
      <group scale={k * 1.02} position={fit.center.clone().multiplyScalar(-k * 1.02)}>
        <primitive object={rim} />
      </group>
    </group>
  );
}
