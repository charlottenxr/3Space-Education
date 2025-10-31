/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, Type, Chat } from '@google/genai';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- Interfaces for Data Structures ---
interface QAPair {
  question: string;
  answer: string;
  answerHtml: string;
  selectionContext: any | null;
}

interface Session {
  id: number;
  timestamp: number;
  modelData: string;
  qaHistory: QAPair[];
  mode: 'geometry' | 'chemistry';
  previewImage: string;
  mimeType: string;
}

// --- DOM Elements ---
const modeSelectionContainer = document.getElementById('mode-selection-container') as HTMLElement;
const appContainer = document.getElementById('app-container') as HTMLElement;
const geometryModeBtn = document.getElementById('geometry-mode-btn') as HTMLButtonElement;
const chemistryModeBtn = document.getElementById('chemistry-mode-btn') as HTMLButtonElement;
const loadSessionBtn = document.getElementById('load-session-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
const saveSessionBtn = document.getElementById('save-session-btn') as HTMLButtonElement;
const appTitle = document.getElementById('app-title') as HTMLElement;
const imageInput = document.getElementById('image-input') as HTMLInputElement;
const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
const canvasContainer = document.getElementById('canvas-container') as HTMLElement;
const codeContainer = document.getElementById('code-container') as HTMLElement;
const loadingContainer = document.getElementById('loading-container') as HTMLElement;
const progressBar = document.getElementById('progress-bar') as HTMLElement;
const loadingText = document.getElementById('loading-text') as HTMLParagraphElement;
const errorMessage = document.getElementById('error-message') as HTMLElement;
const resultsContainer = document.getElementById('results-container') as HTMLElement;
const imagePreview = document.getElementById('image-preview') as HTMLImageElement;
const imagePreviewContainer = document.getElementById('image-preview-container') as HTMLElement;
const detailsContainer = document.getElementById('details-container') as HTMLElement;
const detailsContent = document.getElementById('details-content') as HTMLElement;
const selectionDetailsContainer = document.getElementById('selection-details-container') as HTMLElement;
const selectionDetailsContent = document.getElementById('selection-details-content') as HTMLElement;
const qaContainer = document.getElementById('qa-container') as HTMLElement;
const qaHistoryContainer = document.getElementById('qa-history-container') as HTMLElement;
const qaInput = document.getElementById('qa-input') as HTMLTextAreaElement;
const qaBtn = document.getElementById('qa-btn') as HTMLButtonElement;
const loadModal = document.getElementById('load-modal') as HTMLElement;
const closeModalBtn = document.getElementById('close-modal-btn') as HTMLElement;
const savedSessionsList = document.getElementById('saved-sessions-list') as HTMLElement;
const noSessionsMessage = document.getElementById('no-sessions-message') as HTMLElement;
const notification = document.getElementById('notification') as HTMLElement;


// --- State ---
let currentMode: 'geometry' | 'chemistry' | null = null;
let selectedImage: { data: string; mimeType: string; } | null = null;
let highlightedObjects: { object: THREE.Mesh, originalColor: THREE.Color }[] = [];
let generatedModelData: string | null = null;
let currentSelectionDetails: any | null = null;
let qaHistory: QAPair[] = [];
let notificationTimeout: number | null = null;
let chat: Chat | null = null;
let currentSessionId: number | null = null;


// --- Gemini AI Setup ---
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
const model = 'gemini-2.5-pro';

// --- Schemas for JSON output ---
const geometrySchema = {
    type: Type.OBJECT,
    properties: {
        vertices: {
            type: Type.ARRAY,
            description: "Array of vertex positions, each as an array [<x>, <y>, <z>].",
            items: { type: Type.ARRAY, items: { type: Type.NUMBER } },
        },
        faces: {
            type: Type.ARRAY,
            description: "Array of FACE GROUP objects. Each group represents one original face from the drawing (e.g., a single trapezoid, even if it has a hole).",
            items: {
                type: Type.OBJECT,
                properties: {
                    details: {
                        type: Type.OBJECT,
                        description: "Calculated details for the ENTIRE original face.",
                        properties: {
                            surfaceArea: { type: Type.STRING, description: "Total surface area of this face." },
                            perimeter: { type: Type.STRING, description: "Perimeter of the outer boundary of this face." },
                        },
                         required: ['surfaceArea', 'perimeter'],
                    },
                    triangles: {
                        type: Type.ARRAY,
                        description: "Array of individual triangles that tile to form this entire face.",
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                indices: {
                                    type: Type.ARRAY,
                                    description: "An array of exactly 3 integer indices from the 'vertices' array, defining a single triangle.",
                                    items: { type: Type.INTEGER }
                                },
                            },
                             required: ['indices'],
                        }
                    },
                },
                required: ['details', 'triangles'],
            },
        },
        labels: {
            type: Type.ARRAY,
            description: "Dimension labels shown in the drawing.",
            items: {
                type: Type.OBJECT,
                properties: {
                    text: { type: Type.STRING },
                    position: { type: Type.ARRAY, items: { type: Type.NUMBER } },
                },
                required: ['text', 'position'],
            },
        },
        analysis: {
            type: Type.OBJECT,
            description: "Calculated properties for the whole shape.",
            properties: {
                volume: { type: Type.STRING },
                surfaceArea: { type: Type.STRING },
            },
            required: ['volume', 'surfaceArea'],
        },
    },
    required: ['vertices', 'faces', 'labels', 'analysis'],
};

const chemistrySchema = {
    type: Type.OBJECT,
    properties: {
        atoms: {
            type: Type.ARRAY,
            description: "Array of atom objects.",
            items: {
                type: Type.OBJECT,
                properties: {
                    element: { type: Type.STRING, description: "Chemical symbol (e.g., C, H, O)." },
                    position: { type: Type.ARRAY, description: "3D coordinates [x, y, z].", items: { type: Type.NUMBER } },
                    vseprShape: { type: Type.STRING, description: "VSEPR shape (e.g., Tetrahedral, N/A)." },
                    bondAngles: {
                        type: Type.ARRAY,
                        description: "Bond angles centered on this atom.",
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                angle: { type: Type.STRING, description: "Angle value (e.g., 109.5°)." },
                                atomsInvolved: { type: Type.ARRAY, description: "Symbols of atoms in the angle.", items: { type: Type.STRING } },
                                atomsInvolvedIndices: { type: Type.ARRAY, description: "Indices of atoms in the angle [end1, center, end2].", items: { type: Type.INTEGER } },
                            },
                            required: ['angle', 'atomsInvolved', 'atomsInvolvedIndices'],
                        },
                    },
                },
                required: ['element', 'position', 'vseprShape', 'bondAngles'],
            },
        },
        bonds: {
            type: Type.ARRAY,
            description: "Array of bond objects.",
            items: {
                type: Type.OBJECT,
                properties: {
                    start: { type: Type.INTEGER, description: "Index of the starting atom." },
                    end: { type: Type.INTEGER, description: "Index of the ending atom." },
                    type: { type: Type.STRING, description: "Bond type (e.g., single, double, triple)." },
                    energy: { type: Type.STRING, description: "Bond energy (e.g., 413 kJ/mol)." },
                },
                required: ['start', 'end', 'type', 'energy'],
            },
        },
        analysis: {
            type: Type.OBJECT,
            description: "Overall chemical information.",
            properties: {
                name: { type: Type.STRING, description: "Common chemical name." },
                bondingType: { type: Type.STRING, description: "Primary bonding type (e.g., Covalent)." },
            },
            required: ['name', 'bondingType'],
        },
    },
    required: ['atoms', 'bonds', 'analysis'],
};

// --- Prompts ---
const GEOMETRY_PROMPT = `
You are an expert CAD (Computer-Aided Design) engine. Your task is to analyze the provided 2D technical drawing and generate a precise 3D model in a specific JSON format.

**CRITICAL INSTRUCTIONS:**
1.  **Multi-View Drawing Analysis (Most Important Step):**
    - First, analyze the entire image to identify if it contains multiple views of the same object (e.g., "Figure 1" and "Figure 2", orthographic projections like front/top/side views, or perspective views).
    - You MUST synthesize information from ALL available views to construct a single, coherent 3D model.
    - **Perspective Correction (CRITICAL):** You must actively detect if a view is in perspective. Shapes in perspective are distorted (e.g., a circle appears as an ellipse, a square as a trapezoid). Your primary task is to reverse this distortion. Use orthographic (flat) views to determine the *true shape* of a face, and use perspective views to determine depth and 3D arrangement. For example, if "Figure 1" shows a perfect circle with an 8cm diameter and "Figure 2" shows that face in perspective as an ellipse, you MUST model it as a perfect circle with an 8cm diameter in the final 3D space.
    - Do not merge or flatten different views into a single distorted shape. Treat them as separate pieces of information describing one object. This is the key to correctly interpreting the image.
2.  **Coordinate System & Proportionality:**
    - Use a right-handed coordinate system with the origin [0, 0, 0] at the bottom-left-front corner.
    - Use the explicit dimensions from the drawing to calculate exact vertex coordinates.
    - **Crucially, the final 3D model's dimensions MUST be relatively proportional to the labels in the drawing.** For example, an edge labeled "10cm" must be exactly twice the length of an edge labeled "5cm" in the coordinate space.
3.  **Internal Voids and Holes:**
    - You must model internal holes and voids. This is done by defining the geometry of the inner surfaces.
    - For an object with a hole, you must generate vertices and faces for the outer shell AND the inner walls of the hole.
4.  **Interpret Line Types:** Solid lines are visible foreground edges. Dotted/dashed lines are hidden edges behind solid structures. Use this to determine the 3D form.
5.  **Triangulation and Grouping (VERY IMPORTANT):**
    - The output 'faces' array must contain FACE GROUP objects.
    - Each FACE GROUP represents one original, larger face from the drawing (e.g., the front trapezoid, the top rectangle, the inner wall of the cylinder).
    - Inside each FACE GROUP, you MUST provide a 'triangles' array. This array must contain all the individual, small triangles needed to tile that entire surface.
    - Therefore, the 'indices' array for each triangle object MUST always contain exactly 3 vertex indices.
6.  **Detailed Face Information:** For each FACE GROUP, calculate the total surface area and perimeter of that original face and place it in the 'details' object, with given or inferred parameters. If not enough information is given, say "More information reuqired.".
7.  **Overall Analysis:** Calculate the total volume and total surface area for the entire object.
8.  **Output Format:** Output a single, valid JSON object that strictly adheres to the provided schema. The 'vertices', 'faces', 'labels', and 'analysis' keys are mandatory.
`;


const CHEMISTRY_PROMPT = `
You are an expert computational chemist. Your task is to analyze the provided image of a chemical structure and generate a highly accurate 3D ball-and-stick model in a specific JSON format. Geometric accuracy is the highest priority.

**Core Principle: VSEPR Theory**
- You MUST strictly apply Valence Shell Electron Pair Repulsion (VSEPR) theory to determine the molecule's 3D geometry.
- Electron pairs (both bonding and lone pairs) must be arranged to minimize repulsion, which dictates the precise bond angles and the overall VSEPR shape of each central atom.

**Critical Requirement: Accurate Bond Lengths**
- You MUST use standard, accepted scientific values for bond lengths. Adhere to the following table as a primary reference (1 Å = 1 three.js unit). Interpolate for unlisted but similar bonds.
  - C-H: ~1.09 Å
  - C-C: ~1.54 Å
  - C=C: ~1.34 Å
  - C≡C: ~1.20 Å
  - C-O: ~1.43 Å
  - C=O: ~1.23 Å
  - O-H: ~0.96 Å
- The final coordinates in the JSON must reflect these lengths precisely.

**Instructions:**
1.  **Identify Structure:** Determine the molecule, its atoms, and their connectivity. For aromatic rings like benzene, represent the structure with alternating single and double bonds (a Kekulé structure).
2.  **Apply VSEPR:** For each central atom, determine its VSEPR shape (e.g., "Tetrahedral", "Trigonal Planar") and the ideal bond angles.
3.  **Construct Coordinates:** Build the 3D coordinates based on the VSEPR-derived angles and the standard bond lengths from the table. The goal is the most stable, lowest-energy conformation.
4.  **Provide Details:**
    - For each central atom, list its VSEPR shape and all relevant bond angles, including atom indices.
    - For each bond, specify its type (single, double, triple) and estimated bond energy.
    - Include an "analysis" object with the chemical name and primary bonding type.
5.  **Output Format:** Output a single, valid JSON object adhering strictly to the provided schema.
`;


// --- Three.js Setup ---
let scene: THREE.Scene, camera: THREE.PerspectiveCamera, renderer: THREE.WebGLRenderer, controls: OrbitControls;
let generatedObject: THREE.Object3D | null = null;
let angleVisualizationGroup: THREE.Group;

function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    camera = new THREE.PerspectiveCamera(75, 16 / 9, 0.1, 1000);
    camera.position.z = 5;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    canvasContainer.appendChild(renderer.domElement);
    
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
    directionalLight.position.set(5, 5, 5).normalize();
    scene.add(directionalLight);
    
    angleVisualizationGroup = new THREE.Group();
    scene.add(angleVisualizationGroup);

    renderer.domElement.addEventListener('click', onCanvasClick, false);

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    if (appContainer.classList.contains('hidden')) return;
    const width = canvasContainer.clientWidth;
    const height = canvasContainer.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}
window.addEventListener('resize', onWindowResize, false);


// --- Interactivity ---
function drawAngleVisualization(p1: THREE.Vector3, pCenter: THREE.Vector3, p2: THREE.Vector3, angleText: string) {
    const v1 = new THREE.Vector3().subVectors(p1, pCenter);
    const v2 = new THREE.Vector3().subVectors(p2, pCenter);

    const angle = v1.angleTo(v2);

    const normal = new THREE.Vector3().crossVectors(v1, v2);
    // Use a fallback for 180-degree angles where cross product is zero
    if (normal.lengthSq() < 0.0001) {
        // Find an arbitrary vector not parallel to v1
        let nonParallel = new THREE.Vector3(1, 0, 0);
        if (Math.abs(v1.clone().normalize().dot(nonParallel)) > 0.999) {
            nonParallel = new THREE.Vector3(0, 1, 0);
        }
        normal.crossVectors(v1, nonParallel).normalize();
    } else {
        normal.normalize();
    }
    
    // Don't draw for 0 degrees
    if (angle < 0.01) return;

    const arcRadius = 0.4 * CHEM_MODEL_SCALE;

    // Create the arc geometry
    const curve = new THREE.ArcCurve(0, 0, arcRadius, 0, angle, false);
    const points = curve.getPoints(32);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0xffff99 });
    const arcLine = new THREE.Line(geometry, material);

    // Create a pivot object to orient the arc
    const pivot = new THREE.Object3D();
    pivot.position.copy(pCenter);

    const xAxis = v1.clone().normalize();
    const yAxis = normal.clone().cross(xAxis).normalize();
    const zAxis = normal;

    const matrix = new THREE.Matrix4();
    matrix.makeBasis(xAxis, yAxis, zAxis);
    pivot.quaternion.setFromRotationMatrix(matrix);

    pivot.add(arcLine);
    angleVisualizationGroup.add(pivot);

    // Create and position the label
    const labelSprite = createLabelSprite(angleText);
    if (labelSprite) {
        const bisector = v1.clone().normalize().add(v2.clone().normalize());
        // Handle 180-degree case for bisector
        if (bisector.lengthSq() < 0.0001) {
            bisector.copy(yAxis); // Use the calculated perpendicular axis
        }
        bisector.normalize();
        
        const labelOffset = (arcRadius + 0.15) * 1.5; // Position label relative to arc
        labelSprite.position.copy(pCenter).add(bisector.multiplyScalar(labelOffset));
        labelSprite.scale.set(0.5, 0.5, 0.5);
        angleVisualizationGroup.add(labelSprite);
    }
}

function handleHighlight(intersect: THREE.Intersection) {
    angleVisualizationGroup.clear();

    if (intersect.object instanceof THREE.Mesh && !(intersect.object instanceof THREE.Sprite)) {
        const clickedMesh = intersect.object;

        if (currentMode === 'geometry') {
            const faceId = clickedMesh.userData.faceId;
            if (faceId === undefined) return;

            if(clickedMesh.userData.details) {
                const selectionData = {
                    type: 'geometry_face',
                    id: faceId,
                    details: clickedMesh.userData.details
                };
                displaySelectionDetails(selectionData.details);
                currentSelectionDetails = selectionData;
            }

            generatedObject?.children.forEach(child => {
                if (child instanceof THREE.Mesh && child.userData.faceId === faceId) {
                    const material = child.material as THREE.MeshStandardMaterial;
                    const originalColor = material.color.clone();
                    material.color.setHex(0xffff99); // Light yellow
                    highlightedObjects.push({ object: child, originalColor: originalColor });
                }
            });

        } else if (currentMode === 'chemistry') {
            let details: any;

            // Check if the clicked mesh is part of a bond
            let bondGroup: THREE.Group | null = null;
            let currentObject: THREE.Object3D = clickedMesh;
            // Traverse up to find the main bond group which holds the details
            while(currentObject.parent && !(currentObject.userData.details?.type)){
                currentObject = currentObject.parent;
            }
             if (currentObject.userData.details?.type) {
                bondGroup = currentObject as THREE.Group;
            }

            if (bondGroup) {
                // --- It's a Bond ---
                details = bondGroup.userData.details;
                if (details) {
                    const selectionData = {
                        type: 'chemistry_bond',
                        id: { start: details.startAtomIndex, end: details.endAtomIndex },
                        details: details
                    };
                    displaySelectionDetails(selectionData.details);
                    currentSelectionDetails = selectionData;
                }

                // Highlight all cylinders in the bond group
                bondGroup.traverse(child => {
                    if (child instanceof THREE.Mesh) {
                        const material = child.material as THREE.MeshStandardMaterial;
                        const originalColor = material.color.clone();
                        material.color.setHex(0xffff99);
                        highlightedObjects.push({ object: child, originalColor: originalColor });
                    }
                });
            } else if (clickedMesh.userData.details?.element) {
                // --- It's an Atom ---
                details = clickedMesh.userData.details;
                const selectionData = {
                    type: 'chemistry_atom',
                    id: details.atomIndex,
                    details: details
                };
                displaySelectionDetails(selectionData.details);
                currentSelectionDetails = selectionData;

                // Single object highlight for atom
                const material = clickedMesh.material as THREE.MeshStandardMaterial;
                const originalColor = material.color.clone();
                material.color.setHex(0xffff99);
                highlightedObjects.push({ object: clickedMesh, originalColor: originalColor });

                // If it's an atom, draw its angles
                if (generatedObject && generatedObject.userData.atomPositions) {
                    const originalAtomPositions = generatedObject.userData.atomPositions as THREE.Vector3[];
                    
                    // Apply the parent group's world matrix to get the final, visible positions
                    const finalAtomPositions = originalAtomPositions.map(p => p.clone().applyMatrix4(generatedObject!.matrixWorld));

                    if (details.bondAngles && Array.isArray(details.bondAngles)) {
                        details.bondAngles.forEach((angleInfo: any) => {
                            const indices = angleInfo.atomsInvolvedIndices;
                            if (indices && indices.length === 3) {
                                const p1 = finalAtomPositions[indices[0]];
                                const pCenter = finalAtomPositions[indices[1]];
                                const p2 = finalAtomPositions[indices[2]];

                                if (p1 && pCenter && p2) {
                                    drawAngleVisualization(p1, pCenter, p2, angleInfo.angle);
                                }
                            }
                        });
                    }
                }
            }
        }
    }
}


function onCanvasClick(event: MouseEvent) {
    // 1. Clear previous state
    if (highlightedObjects.length > 0) {
        highlightedObjects.forEach(h => {
            (h.object.material as THREE.MeshStandardMaterial).color.copy(h.originalColor);
        });
        highlightedObjects = [];
    }
    angleVisualizationGroup.clear();
    selectionDetailsContainer.classList.add('hidden');
    selectionDetailsContent.innerHTML = '';
    currentSelectionDetails = null;

    if (!generatedObject) return;

    // 2. Set up Raycaster
    const mouse = new THREE.Vector2();
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);

    // 3. Find intersections
    const intersects = raycaster.intersectObject(generatedObject, true);

    if (intersects.length > 0) {
        handleHighlight(intersects[0]);
    }
}


function frameObject(object: THREE.Object3D) {
    const boundingBox = new THREE.Box3().setFromObject(object);
    const center = boundingBox.getCenter(new THREE.Vector3());
    const size = boundingBox.getSize(new THREE.Vector3());

    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5; 
    
    const direction = controls.object.position.clone().sub(controls.target).normalize();
    camera.position.copy(center).add(direction.multiplyScalar(cameraZ));
    camera.lookAt(center);

    controls.target.copy(center);
    controls.update();
}

function reHighlightSelection(selectionContext: any) {
    if (!generatedObject || !selectionContext) return;

    // 1. Clear any existing highlights
    if (highlightedObjects.length > 0) {
        highlightedObjects.forEach(h => {
            (h.object.material as THREE.MeshStandardMaterial).color.copy(h.originalColor);
        });
        highlightedObjects = [];
    }
    angleVisualizationGroup.clear();
    selectionDetailsContainer.classList.add('hidden');
    selectionDetailsContent.innerHTML = '';
    currentSelectionDetails = null;

    // 2. Perform new highlight based on context
    currentSelectionDetails = selectionContext;
    displaySelectionDetails(selectionContext.details);
    
    switch(selectionContext.type) {
        case 'geometry_face':
            generatedObject?.children.forEach(child => {
                if (child instanceof THREE.Mesh && child.userData.faceId === selectionContext.id) {
                    const material = child.material as THREE.MeshStandardMaterial;
                    const originalColor = material.color.clone();
                    material.color.setHex(0xffff99); // Light yellow
                    highlightedObjects.push({ object: child, originalColor: originalColor });
                }
            });
            break;
        
        case 'chemistry_atom':
            generatedObject.traverse(child => {
                if (child instanceof THREE.Mesh && child.userData.details?.atomIndex === selectionContext.id) {
                    const material = child.material as THREE.MeshStandardMaterial;
                    const originalColor = material.color.clone();
                    material.color.setHex(0xffff99);
                    highlightedObjects.push({ object: child, originalColor: originalColor });

                    if (generatedObject && generatedObject.userData.atomPositions) {
                        const originalAtomPositions = generatedObject.userData.atomPositions as THREE.Vector3[];
                        const finalAtomPositions = originalAtomPositions.map(p => p.clone().applyMatrix4(generatedObject!.matrixWorld));
                        const details = selectionContext.details;

                        if (details.bondAngles && Array.isArray(details.bondAngles)) {
                            details.bondAngles.forEach((angleInfo: any) => {
                                const indices = angleInfo.atomsInvolvedIndices;
                                if (indices && indices.length === 3) {
                                    const p1 = finalAtomPositions[indices[0]];
                                    const pCenter = finalAtomPositions[indices[1]];
                                    const p2 = finalAtomPositions[indices[2]];
                                    if (p1 && pCenter && p2) {
                                        drawAngleVisualization(p1, pCenter, p2, angleInfo.angle);
                                    }
                                }
                            });
                        }
                    }
                }
            });
            break;

        case 'chemistry_bond':
             generatedObject.traverse(child => {
                if (child instanceof THREE.Group && child.userData.details?.startAtomIndex === selectionContext.id.start && child.userData.details?.endAtomIndex === selectionContext.id.end) {
                    child.traverse(mesh => {
                         if (mesh instanceof THREE.Mesh) {
                            const material = mesh.material as THREE.MeshStandardMaterial;
                            const originalColor = material.color.clone();
                            material.color.setHex(0xffff99);
                            highlightedObjects.push({ object: mesh, originalColor: originalColor });
                        }
                    });
                }
            });
            break;
    }
}

// --- Helper Functions for AI ---

/**
 * Converts a response string with potential Markdown into formatted HTML.
 * @param text The raw response string.
 * @returns An HTML string.
 */
function formatResponseTextToHtml(text: string): string {
    // 1. Strip any potential LaTeX delimiters just in case.
    let processedText = text.replace(/\$\$([\s\S]*?)\$\$|\$([\s\S]*?)\$/g, (match, displayMath, inlineMath) => {
        return displayMath || inlineMath || '';
    });

    // 2. Apply custom math formatting for superscripts, subscripts, etc.
    processedText = formatMathString(processedText);

    // 3. Convert simple Markdown to HTML tags.
    processedText = processedText
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/\*(.*?)\*/g, '<em>$1</em>')         // Italic
        .replace(/`([^`]+)`/g, '<code>$1</code>');      // Inline code

    // 4. Convert newlines to <br> for paragraph breaks.
    processedText = processedText.replace(/\n/g, '<br />');

    return processedText;
}

function formatMathString(input: any): string {
    if (typeof input !== 'string') {
        return String(input);
    }

    const superscripts: { [key: string]: string } = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ', 'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ',
        'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ', 'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ', 'u': 'ᵘ',
        'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
        'A': 'ᴬ', 'B': 'ᴮ', 'C': 'ᶜ', 'D': 'ᴰ', 'E': 'ᴱ', 'F': 'ᶠ', 'G': 'ᴳ', 'H': 'ᴴ', 'I': 'ᴵ', 'J': 'ᴶ', 
        'K': 'ᴷ', 'L': 'ᴸ', 'M': 'ᴹ', 'N': 'ᴺ', 'O': 'ᴼ', 'P': 'ᴾ', 'R': 'ᴿ', 'T': 'ᵀ', 'U': 'ᵁ', 'V': 'ⱽ', 'W': 'ᵂ',
        'X': 'ᵡ', 'Y': 'ʸ', 'Z': 'ᶻ',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
    };
    
    const subscripts: { [key: string]: string } = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ', 'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
        'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ', 'v': 'ᵥ', 'x': 'ₓ',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎'
    };

    const toSuper = (chars: string) => chars.split('').map(c => superscripts[c.toLowerCase()] || c).join('');
    const toSub = (chars: string) => chars.split('').map(c => subscripts[c.toLowerCase()] || c).join('');

    let formatted = input;

    // Replace sqrt keyword with symbol
    formatted = formatted.replace(/sqrt/gi, '√');

    // Replace common fractions
    formatted = formatted.replace(/1\/2/g, '½').replace(/1\/4/g, '¼').replace(/3\/4/g, '¾').replace(/1\/3/g, '⅓').replace(/2\/3/g, '⅔');

    // Handle grouped superscripts like ^(text) or ^{text}
    formatted = formatted.replace(/\^[\({]([^)}]+)[\)}]/g, (_, content) => toSuper(content));
    // Handle single character superscripts like ^2 or ^n
    formatted = formatted.replace(/\^([a-zA-Z0-9+\-=()])/g, (_, char) => toSuper(char));

    // Handle grouped subscripts like _(text) or _{text}
    formatted = formatted.replace(/_[\({]([^)}]+)[\)}]/g, (_, content) => toSub(content));
    // Handle single character subscripts like _2 or _n
    formatted = formatted.replace(/_([a-zA-Z0-9+\-=()])/g, (_, char) => toSub(char));

    return formatted;
}


function createLabelSprite(text: string) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    if (!context) return null;

    const fontSize = 90;
    const font = `Bold ${fontSize}px 'Roboto', sans-serif`;
    context.font = font;
    const metrics = context.measureText(text);
    const textWidth = metrics.width;
    canvas.width = textWidth;
    canvas.height = fontSize;
    context.font = font;
    context.fillStyle = 'white';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, textWidth / 2, fontSize / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(1.0 * (textWidth / fontSize), 1.0, 1.0);
    return sprite;
}

const CHEM_MODEL_SCALE = 3.0;
const CPK_COLORS: { [key: string]: number } = { H: 0xffffff, C: 0x808080, N: 0x0000ff, O: 0xff0000, F: 0x00ff00, CL: 0x00ff00, BR: 0xa52a2a, I: 0x800080, P: 0xffa500, S: 0xffff00, B: 0xfa8072, DEFAULT: 0xffc0cb };
const ATOM_RADII: { [key: string]: number } = { 
    H: 0.2 * CHEM_MODEL_SCALE, C: 0.35 * CHEM_MODEL_SCALE, N: 0.3 * CHEM_MODEL_SCALE, O: 0.3 * CHEM_MODEL_SCALE, F: 0.25 * CHEM_MODEL_SCALE, CL: 0.5 * CHEM_MODEL_SCALE, BR: 0.55 * CHEM_MODEL_SCALE, I: 0.65 * CHEM_MODEL_SCALE, P: 0.5 * CHEM_MODEL_SCALE, S: 0.5 * CHEM_MODEL_SCALE, B: 0.4 * CHEM_MODEL_SCALE, DEFAULT: 0.25 * CHEM_MODEL_SCALE 
};

function createAtom(atomData: any, position: THREE.Vector3, index: number) {
    const group = new THREE.Group();
    const el = atomData.element.toUpperCase();
    const color = CPK_COLORS[el] || CPK_COLORS['DEFAULT'];
    const radius = ATOM_RADII[el] || ATOM_RADII['DEFAULT'];

    const sphereGeometry = new THREE.SphereGeometry(radius, 32, 16);
    const sphereMaterial = new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.2 });
    const atomMesh = new THREE.Mesh(sphereGeometry, sphereMaterial);
    atomMesh.userData.details = {
        element: atomData.element,
        bondAngles: atomData.bondAngles,
        vseprShape: atomData.vseprShape,
        atomIndex: index,
    };
    group.add(atomMesh);
    
    const label = createLabelSprite(atomData.element);
    if (label) {
        label.position.y += radius + (0.15 * CHEM_MODEL_SCALE); // Offset label above the atom
        label.scale.multiplyScalar(radius * 2.5); // Scale label with atom size
        group.add(label);
    }
    
    group.position.copy(position);
    return group;
}

function createBond(bondData: any, startPos: THREE.Vector3, endPos: THREE.Vector3, bondName: string): THREE.Group {
    const bondGroup = new THREE.Group();
    bondGroup.userData.details = {
        name: bondName,
        type: bondData.type,
        energy: bondData.energy,
        startAtomIndex: bondData.start,
        endAtomIndex: bondData.end,
    };

    const path = new THREE.Vector3().subVectors(endPos, startPos);
    const length = path.length();
    const baseMaterial = new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.5, metalness: 0.2 });
    const bondRadius = 0.05 * CHEM_MODEL_SCALE;

    const createCylinder = () => {
        const cylinderGeometry = new THREE.CylinderGeometry(bondRadius, bondRadius, length, 16);
        return new THREE.Mesh(cylinderGeometry, baseMaterial.clone()); // CLONE material here
    };

    const cylinders: THREE.Mesh[] = [];
    const bondType = bondData.type.toLowerCase();
    
    if (bondType === 'double') {
        cylinders.push(createCylinder(), createCylinder());
    } else if (bondType === 'triple') {
        cylinders.push(createCylinder(), createCylinder(), createCylinder());
    } else { // Single bond
        cylinders.push(createCylinder());
    }

    if (cylinders.length > 1) {
        let offsetDirection = new THREE.Vector3().crossVectors(path, new THREE.Vector3(0, 1, 0));
        if (offsetDirection.length() < 0.01) { // Handle case where bond is parallel to Y-axis
            offsetDirection = new THREE.Vector3().crossVectors(path, new THREE.Vector3(1, 0, 0));
        }
        offsetDirection.normalize().multiplyScalar(bondRadius * 1.5);

        if (cylinders.length === 2) {
             cylinders[0].position.add(offsetDirection);
             cylinders[1].position.sub(offsetDirection);
        } else if (cylinders.length === 3) {
            cylinders[1].position.add(offsetDirection);
            cylinders[2].position.sub(offsetDirection);
        }
    }

    cylinders.forEach(cylinder => {
        const cylinderGroup = new THREE.Group();
        cylinderGroup.position.copy(startPos).add(path.clone().multiplyScalar(0.5));
        cylinderGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), path.clone().normalize());
        cylinderGroup.add(cylinder);
        bondGroup.add(cylinderGroup);
    });

    return bondGroup;
}


// --- Model Builders from JSON ---
function buildGeometryModel(data: any): THREE.Group {
    const group = new THREE.Group();

    if (!data.vertices || !Array.isArray(data.vertices) || data.vertices.length === 0) {
        throw new Error("Invalid JSON structure: 'vertices' array not found or is empty.");
    }
    if (!data.faces || !Array.isArray(data.faces) || data.faces.length === 0) {
        throw new Error("Invalid JSON structure: 'faces' array not found or is empty.");
    }

    const baseFaceMaterial = new THREE.MeshStandardMaterial({
        color: 0x03dac6, side: THREE.DoubleSide, flatShading: false,
        metalness: 0.2, roughness: 0.7, polygonOffset: true,
        polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });

    const allVertices = data.vertices;

    data.faces.forEach((faceGroup: any, faceIndex: number) => {
        if (!faceGroup.triangles || !Array.isArray(faceGroup.triangles)) return;

        faceGroup.triangles.forEach((triangleData: any) => {
            const triangleIndices = triangleData.indices;
            if (!triangleIndices || !Array.isArray(triangleIndices) || triangleIndices.length !== 3) {
                 console.warn('Skipping triangle with invalid indices:', triangleData);
                return;
            }

            // Validate vertices to prevent NaN errors in Three.js
            const faceVerticesArrays = triangleIndices.map((index: number) => allVertices[index]);
            const hasInvalidVertex = faceVerticesArrays.some(v => 
                !v || !Array.isArray(v) || v.length !== 3 || v.some(coord => typeof coord !== 'number' || !isFinite(coord))
            );

            if (hasInvalidVertex) {
                console.warn('Skipping triangle with invalid vertex data:', faceVerticesArrays);
                return; // Prevent crash by skipping this triangle
            }

            const faceGeometry = new THREE.BufferGeometry();
            const positions = new Float32Array(faceVerticesArrays.flat());
            faceGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            faceGeometry.setIndex([0, 1, 2]);
            faceGeometry.computeVertexNormals();

            const faceMesh = new THREE.Mesh(faceGeometry, baseFaceMaterial.clone());
            faceMesh.userData.details = faceGroup.details; // Details for the whole face
            faceMesh.userData.faceId = faceIndex; // Group identifier
            group.add(faceMesh);
        });
    });


    if (group.children.filter(c => c.type === "Mesh").length === 0) {
         throw new Error("Could not create any valid geometric shapes from the provided JSON.");
    }

    if (data.labels && Array.isArray(data.labels)) {
        data.labels.forEach((labelData: any) => {
            if (labelData.text && Array.isArray(labelData.position)) {
                const labelSprite = createLabelSprite(labelData.text);
                if (labelSprite) {
                    labelSprite.position.fromArray(labelData.position);
                    group.add(labelSprite);
                }
            }
        });
    }

    // Dynamically scale labels based on overall model size
    const modelBox = new THREE.Box3().setFromObject(group);
    const modelSize = modelBox.getSize(new THREE.Vector3());
    const maxDim = Math.max(modelSize.x, modelSize.y, modelSize.z);

    if (maxDim > 0.001) {
        const labelScale = maxDim / 15; // Tuned for geometry
        group.children.forEach(child => {
            if (child instanceof THREE.Sprite) {
                child.scale.multiplyScalar(labelScale);
            }
        });
    }

    return group;
}


function buildChemistryModel(data: any): THREE.Group {
    const group = new THREE.Group();
    if (!data.atoms || !Array.isArray(data.atoms)) throw new Error("Invalid JSON: 'atoms' array not found.");
    if (data.atoms.length === 0) throw new Error("Invalid JSON: 'atoms' array cannot be empty.");

    const atomPositions: THREE.Vector3[] = [];
    data.atoms.forEach((atom: any, index: number) => {
        const position = new THREE.Vector3().fromArray(atom.position || [0, 0, 0]).multiplyScalar(CHEM_MODEL_SCALE);
        atomPositions.push(position);
        const atomGroup = createAtom(atom, position, index);
        group.add(atomGroup);
    });

    if (data.bonds && Array.isArray(data.bonds)) {
        data.bonds.forEach((bond: any) => {
            if (bond.start >= 0 && bond.start < atomPositions.length &&
                bond.end >= 0 && bond.end < atomPositions.length) {
                const startPos = atomPositions[bond.start];
                const endPos = atomPositions[bond.end];
                const startElement = data.atoms[bond.start].element;
                const endElement = data.atoms[bond.end].element;
                const bondName = `${startElement}-${endElement}`;
                const bondGroup = createBond(bond, startPos, endPos, bondName);
                group.add(bondGroup);
            }
        });
    }

    group.userData.atomPositions = atomPositions;
    
    // Dynamically scale labels based on overall model size for readability
    const modelBox = new THREE.Box3().setFromObject(group);
    const modelSize = modelBox.getSize(new THREE.Vector3());
    const maxDim = Math.max(modelSize.x, modelSize.y, modelSize.z);

    if (maxDim > 0.001) {
        const labelScale = maxDim / 30; // Tuned for chemistry
        group.traverse(child => {
            if (child instanceof THREE.Sprite) {
                child.scale.multiplyScalar(labelScale);
            }
        });
    }

    return group;
}

// --- Main Application Logic ---

function selectMode(mode: 'geometry' | 'chemistry') {
    currentMode = mode;
    modeSelectionContainer.classList.add('hidden');
    appContainer.classList.remove('hidden');
    appTitle.textContent = `AI 3D ${mode.charAt(0).toUpperCase() + mode.slice(1)} Generator`;
    saveSessionBtn.classList.add('hidden');

    if (mode === 'geometry') {
        qaInput.placeholder = "eg. how to calculate the volume?";
    } else { // 'chemistry'
        qaInput.placeholder = "what is the shape of the chemical structure around the C atom?";
    }

    onWindowResize();
}

function resetApp() {
    currentMode = null;
    selectedImage = null;
    generatedModelData = null;
    currentSelectionDetails = null;
    qaHistory = [];
    chat = null;
    currentSessionId = null;
    
    imageInput.value = '';
    imagePreviewContainer.classList.add('hidden');
    generateBtn.disabled = true;
    resultsContainer.classList.add('hidden');
    detailsContainer.classList.add('hidden');
    detailsContent.innerHTML = '';
    selectionDetailsContainer.classList.add('hidden');
    selectionDetailsContent.innerHTML = '';
    qaContainer.classList.add('hidden');
    qaHistoryContainer.innerHTML = '';
    qaInput.value = '';
    saveSessionBtn.classList.add('hidden');
    
    clearError();

    if (generatedObject) {
        scene.remove(generatedObject);
        generatedObject = null;
    }

    if (highlightedObjects.length > 0) {
        highlightedObjects.forEach(h => {
            (h.object.material as THREE.MeshStandardMaterial).color.copy(h.originalColor);
        });
        highlightedObjects = [];
    }
    
    angleVisualizationGroup.clear();
    modeSelectionContainer.classList.remove('hidden');
    appContainer.classList.add('hidden');
    checkSavedSessions();
}

function handleImageUpload(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) {
        selectedImage = null;
        generateBtn.disabled = true;
        imagePreviewContainer.classList.add('hidden');
        return;
    }

    if (!['image/jpeg', 'image/png'].includes(file.type)) {
        showError('Please upload a valid JPG or PNG image.');
        return;
    }
    clearError();

    const reader = new FileReader();
    reader.onload = (e) => {
        const result = e.target?.result as string;
        imagePreview.src = result;
        imagePreviewContainer.classList.remove('hidden');
        
        selectedImage = { data: result.split(',')[1], mimeType: file.type };
        generateBtn.disabled = false;
    };
    reader.onerror = () => {
        showError('Failed to read the image file.');
        selectedImage = null;
        generateBtn.disabled = true;
        imagePreviewContainer.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function getChatPrompt(modelData: string) {
    return `
You are an expert educator and tutor for secondary school students, with deep knowledge in chemistry and geometry. A 3D model has been generated, and its underlying data is provided below.
Your task is to answer user questions about this model. Use the model's data as the primary context for your calculations, but explain everything in simple, intuitive terms.

**CRITICAL RULE: Speak like a helpful teacher, not a computer program.**
- **NEVER** mention technical computer graphics terms like "vertices", "indices", "faces", "meshes", "coordinates", or "triangulation". The user does not know what these are.
- Instead, describe the model using familiar geometric shapes (e.g., "a rectangular block", "a hollow cylinder", "a flat circular face").
- Refer to dimensions with common terms (e.g., "length", "width", "height", "radius", "diameter", "thickness").
- Your primary goal is to make the 3D model understandable to a 14-year-old student. Be clear, encouraging, and focus on the concepts, not the raw data.

**Answering Style:**
- When a user asks "how to calculate..." something, provide a step-by-step explanation using the relevant formula and plugging in the dimensions from the model data.
- For example, instead of saying 'the volume is in the analysis object', explain HOW that volume was calculated (e.g., "The model is a rectangular prism. To find its volume, we multiply its length (14 cm) by its width (12 cm) by its height (30 cm)...").

**Formatting Rules:**
- Use markdown for emphasis where needed, such as **bold** for key terms or *italics*.
- For superscripts, use the caret symbol \`^\`. For single characters, use it directly (e.g., \`r^2\`). For multiple characters, group them in parentheses (e.g., \`10^(-3)\`).
- For subscripts, use the underscore symbol \`_\`. For single characters, use it directly (e.g., \`H_2O\` should be written as \`H_2O\`). For multiple characters, group them in parentheses (e.g., \`rate_(initial)\`).
- Use the \`sqrt\` keyword for square roots, like \`sqrt(b^2-4ac)\`.

Your responses should foster a positive learning experience, encouraging curiosity and further questions from the students.

OVERALL MODEL DATA:
---
${modelData}
---
`;
}

function updateLoadingProgress(progress: number, text: string) {
    progressBar.style.width = `${progress}%`;
    loadingText.textContent = text;
}

async function handleGenerateClick() {
    if (!selectedImage || !currentMode) {
        showError('An error occurred. Please select a mode and upload an image.');
        return;
    }

    setLoading(true);
    clearError();
    
    if (generatedObject) {
        scene.remove(generatedObject);
        generatedObject = null;
    }
    
    // Reset states for new generation
    qaInput.value = '';
    qaBtn.disabled = true;
    qaHistory = [];
    qaHistoryContainer.innerHTML = '';
    generatedModelData = null;
    currentSelectionDetails = null;
    chat = null;
    currentSessionId = null;
    angleVisualizationGroup.clear();
    detailsContainer.classList.add('hidden');
    detailsContent.innerHTML = '';
    selectionDetailsContainer.classList.add('hidden');
    selectionDetailsContent.innerHTML = '';
    qaContainer.classList.add('hidden');
    saveSessionBtn.classList.add('hidden');

    let slowProgressInterval: number | null = null;

    try {
        updateLoadingProgress(5, 'Preparing request...');
        
        const prompt = currentMode === 'geometry' ? GEOMETRY_PROMPT : CHEMISTRY_PROMPT;
        const schema = currentMode === 'geometry' ? geometrySchema : chemistrySchema;
        
        const textPart = { text: prompt };
        const imagePart = { inlineData: { mimeType: selectedImage.mimeType, data: selectedImage.data } };
        
        const config: any = {
            responseMimeType: "application/json",
            responseSchema: schema,
        };

        if (currentMode === 'geometry') {
            config.thinkingConfig = { thinkingBudget: 8192 };
        }

        await new Promise(resolve => setTimeout(resolve, 200)); // Let user see the first message
        updateLoadingProgress(10, 'Sending to Gemini for analysis...');
        
        const responsePromise = ai.models.generateContent({
            model: model,
            contents: [{ parts: [textPart, imagePart] }],
            config,
        });

        // While waiting, slowly move the bar to show progress
        let progress = 10;
        slowProgressInterval = window.setInterval(() => {
            if (progress < 70) {
                progress += 1;
                progressBar.style.width = `${progress}%`;
            } else {
                if (slowProgressInterval) clearInterval(slowProgressInterval);
                slowProgressInterval = null;
            }
        }, 250);

        const response = await responsePromise;
        if(slowProgressInterval) {
            clearInterval(slowProgressInterval);
            slowProgressInterval = null;
        }

        updateLoadingProgress(75, 'Received response, parsing data...');

        const rawResponseText = response.text;
        if (!rawResponseText || rawResponseText.trim() === '') {
            throw new Error(response.promptFeedback?.blockReason
                ? `Response blocked for safety reasons: ${response.promptFeedback.blockReason}.`
                : "The AI returned an empty response. It might not have interpreted the image."
            );
        }
        
        const jsonString = rawResponseText;
        generatedModelData = jsonString; // Store data for QA

        // Initialize a new chat session with the model data as context
        const chatPrompt = getChatPrompt(generatedModelData);
        chat = ai.chats.create({
            model: 'gemini-2.5-flash',
            history: [
                { role: 'user', parts: [{ text: chatPrompt }] },
                { role: 'model', parts: [{ text: 'Understood. I am ready to answer questions about the provided 3D model data.' }] }
            ],
        });

        qaBtn.disabled = false; // Enable asking questions
        saveSessionBtn.classList.remove('hidden');

        displayCode(jsonString);
        
        const data = JSON.parse(jsonString);

        await new Promise(resolve => setTimeout(resolve, 200));
        updateLoadingProgress(85, 'Constructing 3D model...');
        let resultObject;

        if (currentMode === 'geometry') {
            resultObject = buildGeometryModel(data);
        } else {
            resultObject = buildChemistryModel(data);
        }

        if (data.analysis) {
            displayDetails(data.analysis);
        }

        const isModelValid = (obj: THREE.Object3D) => {
            if (!obj) return false;
            if (obj instanceof THREE.Mesh) return true;
            if (obj instanceof THREE.Group) return obj.children.length > 0;
            return false;
        };

        if (isModelValid(resultObject)) {
            const box = new THREE.Box3().setFromObject(resultObject);
            const center = box.getCenter(new THREE.Vector3());
            resultObject.position.sub(center);

            scene.add(resultObject);
            generatedObject = resultObject;
            frameObject(generatedObject);
        } else {
             throw new Error("Generated JSON did not produce a valid 3D model.");
        }
        
        await new Promise(resolve => setTimeout(resolve, 200));
        updateLoadingProgress(95, 'Finalizing scene...');

        resultsContainer.classList.remove('hidden');
        qaContainer.classList.remove('hidden'); // Show Q&A panel with results
        onWindowResize();

        // If successful, we reach here.
        updateLoadingProgress(100, 'Done!');
        setTimeout(() => setLoading(false), 500);

    } catch (error) {
        if(slowProgressInterval) {
            clearInterval(slowProgressInterval);
        }
        console.error('Error generating 3D model:', error);
        const message = error instanceof Error ? error.message : 'Failed to generate the 3D model. Please try again.';
        showError(message);
        resultsContainer.classList.add('hidden');
        detailsContainer.classList.add('hidden');
        setLoading(false); // On error, just hide the loading screen.
    }
}

async function handleAskQuestion() {
    const question = qaInput.value.trim();
    if (!question || !chat) {
        return;
    }

    qaBtn.disabled = true;
    const thinkingIndicator = document.createElement('div');
    thinkingIndicator.className = 'qa-pair';
    thinkingIndicator.innerHTML = `<div class="qa-question">${question}</div><div class="qa-answer">Thinking...</div>`;
    qaHistoryContainer.prepend(thinkingIndicator);
    qaHistoryContainer.scrollTop = 0;

    try {
        let fullQuestion = question;
        if (currentSelectionDetails) {
            const selectionContext = `\n\n(Note: I have the following part of the model currently selected, please use this as additional context for my question: ${JSON.stringify(currentSelectionDetails.details)})`;
            fullQuestion += selectionContext;
        }

        const response = await chat.sendMessage({ message: fullQuestion });

        const answerText = response.text;
        const answerHtml = formatResponseTextToHtml(answerText);
       
        qaHistory.unshift({ question, answer: answerText, answerHtml: answerHtml, selectionContext: currentSelectionDetails });
        renderQaHistory();
        qaInput.value = '';

    } catch (error) {
        console.error('Error asking question:', error);
        const errorAnswer = 'Sorry, I was unable to answer that question.';
        qaHistory.unshift({ question, answer: errorAnswer, answerHtml: errorAnswer, selectionContext: null });
        renderQaHistory();
    } finally {
        qaBtn.disabled = false;
        qaInput.focus();
    }
}

function renderQaHistory() {
    qaHistoryContainer.innerHTML = '';
    qaHistory.forEach(pair => {
        const pairDiv = document.createElement('div');
        pairDiv.className = 'qa-pair';
        
        const questionDiv = document.createElement('div');
        questionDiv.className = 'qa-question';
        questionDiv.textContent = pair.question;

        if (pair.selectionContext) {
            questionDiv.classList.add('clickable-question');
            questionDiv.title = "Click to re-select this part of the model";
            questionDiv.addEventListener('click', () => {
                reHighlightSelection(pair.selectionContext);
            });
        }

        const answerDiv = document.createElement('div');
        answerDiv.className = 'qa-answer';
        answerDiv.innerHTML = pair.answerHtml;

        pairDiv.appendChild(questionDiv);
        pairDiv.appendChild(answerDiv);
        qaHistoryContainer.appendChild(pairDiv);
    });
    qaHistoryContainer.scrollTop = 0;
}


function displayDetails(analysis: any) {
    detailsContent.innerHTML = '';
    const ul = document.createElement('ul');

    if (currentMode === 'geometry') {
        const volume = formatMathString(analysis.volume || 'N/A');
        const surfaceArea = formatMathString(analysis.surfaceArea || 'N/A');
        ul.innerHTML = `
            <li><strong>Volume:</strong> <span>${volume}</span></li>
            <li><strong>Surface Area:</strong> <span>${surfaceArea}</span></li>
        `;
    } else if (currentMode === 'chemistry') {
        const name = formatMathString(analysis.name || 'N/A');
        const bondingType = formatMathString(analysis.bondingType || 'N/A');
        ul.innerHTML = `
            <li><strong>Chemical Name:</strong> <span>${name}</span></li>
            <li><strong>Bonding Type:</strong> <span>${bondingType}</span></li>
        `;
    }

    if (ul.children.length > 0) {
        detailsContent.appendChild(ul);
        detailsContainer.classList.remove('hidden');
    }
}

function displaySelectionDetails(details: any) {
    if (!details) {
        selectionDetailsContainer.classList.add('hidden');
        selectionDetailsContent.innerHTML = '';
        return;
    }
    selectionDetailsContent.innerHTML = '';
    const ul = document.createElement('ul');

    // Geometry Face Details
    if (details.surfaceArea) {
        ul.innerHTML = `
            <li><strong>Surface Area:</strong> <span>${formatMathString(details.surfaceArea)}</span></li>
            <li><strong>Perimeter:</strong> <span>${formatMathString(details.perimeter || 'N/A')}</span></li>
        `;
    }
    // Chemistry Atom Details
    else if (details.element) {
        let anglesHtml = (details.bondAngles || [])
            .map((angle: any) => `<li>${formatMathString(angle.atomsInvolved.join('-'))}: ${formatMathString(angle.angle)}</li>`)
            .join('');
        ul.innerHTML = `
            <li><strong>Element:</strong> <span>${formatMathString(details.element)}</span></li>
            <li><strong>VSEPR Shape:</strong> <span>${formatMathString(details.vseprShape || 'N/A')}</span></li>
            ${anglesHtml ? `<li><strong>Bond Angles:</strong><ul class="nested-list">${anglesHtml}</ul></li>` : ''}
        `;
    }
    // Chemistry Bond Details
    else if (details.type) {
         ul.innerHTML = `
            <li><strong>Bond Name:</strong> <span>${formatMathString(details.name || 'N/A')}</span></li>
            <li><strong>Bond Type:</strong> <span>${formatMathString(details.type)}</span></li>
            <li><strong>Bond Energy:</strong> <span>${formatMathString(details.energy)}</span></li>
        `;
    }


    if (ul.children.length > 0) {
        selectionDetailsContent.appendChild(ul);
        selectionDetailsContainer.classList.remove('hidden');
    }
}


function displayCode(code: string) {
    try {
        const parsed = JSON.parse(code);
        codeContainer.textContent = JSON.stringify(parsed, null, 2);
    } catch {
        codeContainer.textContent = code;
    }
}

function setLoading(isLoading: boolean) {
    generateBtn.disabled = isLoading;
    if (isLoading) {
        loadingContainer.classList.remove('hidden');
        progressBar.style.width = '0%';
        loadingText.textContent = 'Generating 3D model, this may take a moment...'; // Reset text
        resultsContainer.classList.add('hidden');
        detailsContainer.classList.add('hidden');
        selectionDetailsContainer.classList.add('hidden');
    } else {
        loadingContainer.classList.add('hidden');
    }
}

function showError(message: string) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function clearError() {
    errorMessage.classList.add('hidden');
    errorMessage.textContent = '';
}

// --- Session Management ---
const SESSIONS_KEY = 'ai3d_sessions';

function getSavedSessions(): Session[] {
    try {
        const saved = localStorage.getItem(SESSIONS_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch (e) {
        console.error("Failed to parse saved sessions:", e);
        return [];
    }
}

function showNotification(message: string) {
    if (notificationTimeout) {
        clearTimeout(notificationTimeout);
    }
    notification.textContent = message;
    notification.classList.add('show');
    notificationTimeout = window.setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

function handleSaveSession() {
    if (!generatedModelData || !selectedImage || !currentMode) {
        showNotification('Error: Cannot save session, model data is missing.');
        return;
    }

    let sessions = getSavedSessions();
    const sessionToSave: Omit<Session, 'id' | 'timestamp'> & { timestamp: number } = {
        timestamp: Date.now(),
        modelData: generatedModelData,
        qaHistory: qaHistory,
        mode: currentMode,
        previewImage: selectedImage.data,
        mimeType: selectedImage.mimeType,
    };

    if (currentSessionId) {
        // Update existing session
        const sessionIndex = sessions.findIndex(s => s.id === currentSessionId);
        if (sessionIndex !== -1) {
            sessions[sessionIndex] = { ...sessionToSave, id: currentSessionId };
            // Sort by timestamp to bring the updated one to the top of the list view
            sessions.sort((a, b) => b.timestamp - a.timestamp);
            localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
            showNotification('Session updated successfully!');
        } else {
            // ID exists in state but not in storage, something is wrong. Fallback to new save.
            currentSessionId = null;
        }
    } 
    
    if (!currentSessionId) {
        // Create new session
        const newId = Date.now();
        currentSessionId = newId;
        const newSession: Session = { ...sessionToSave, id: newId, timestamp: newId };
        sessions.unshift(newSession); // Add to front
        localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
        showNotification('Session saved successfully!');
    }

    checkSavedSessions();
}

function handleOpenLoadModal() {
    const sessions = getSavedSessions();
    savedSessionsList.innerHTML = '';

    if (sessions.length === 0) {
        noSessionsMessage.classList.remove('hidden');
    } else {
        noSessionsMessage.classList.add('hidden');
        sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = 'session-item';
            item.innerHTML = `
                <img src="data:${session.mimeType};base64,${session.previewImage}" alt="Session preview">
                <span class="timestamp">${new Date(session.timestamp).toLocaleString()}</span>
            `;
            item.addEventListener('click', () => {
                loadSessionState(session);
                loadModal.classList.add('hidden');
            });
            savedSessionsList.appendChild(item);
        });
    }
    loadModal.classList.remove('hidden');
}

function loadSessionState(session: Session) {
    resetApp();
    currentSessionId = session.id;
    currentMode = session.mode;
    generatedModelData = session.modelData;
    qaHistory = session.qaHistory;
    selectedImage = { data: session.previewImage, mimeType: session.mimeType };

    // --- Restore UI State ---
    modeSelectionContainer.classList.add('hidden');
    appContainer.classList.remove('hidden');
    appTitle.textContent = `AI 3D ${session.mode.charAt(0).toUpperCase() + session.mode.slice(1)} Generator`;
    
    imagePreview.src = `data:${session.mimeType};base64,${session.previewImage}`;
    imagePreviewContainer.classList.remove('hidden');
    
    displayCode(session.modelData);
    renderQaHistory();

    // Re-initialize the chat session from the saved model data and history
    const chatPrompt = getChatPrompt(generatedModelData);
    const chatHistoryPayload = [
        { role: 'user', parts: [{ text: chatPrompt }] },
        { role: 'model', parts: [{ text: 'Understood. I am ready to answer questions about the provided 3D model data.' }] }
    ];

    session.qaHistory.slice().reverse().forEach(pair => {
        chatHistoryPayload.push({ role: 'user', parts: [{ text: pair.question }] });
        chatHistoryPayload.push({ role: 'model', parts: [{ text: pair.answer }] });
    });

    chat = ai.chats.create({
        model: 'gemini-2.5-flash',
        history: chatHistoryPayload
    });

    try {
        const data = JSON.parse(session.modelData);
        let resultObject;
        if (session.mode === 'geometry') {
            resultObject = buildGeometryModel(data);
        } else {
            resultObject = buildChemistryModel(data);
        }

        if (data.analysis) {
            displayDetails(data.analysis);
        }
        
        const box = new THREE.Box3().setFromObject(resultObject);
        const center = box.getCenter(new THREE.Vector3());
        resultObject.position.sub(center);

        scene.add(resultObject);
        generatedObject = resultObject;
        frameObject(generatedObject);
        
        resultsContainer.classList.remove('hidden');
        qaContainer.classList.remove('hidden');
        qaBtn.disabled = false;
        saveSessionBtn.classList.remove('hidden');
        onWindowResize();

    } catch (error) {
        console.error('Error loading session state:', error);
        showError('Failed to load the 3D model from saved data.');
        resetApp();
    }
}

function checkSavedSessions() {
    const sessions = getSavedSessions();
    loadSessionBtn.disabled = sessions.length === 0;
}

// --- Initialization ---
function initializeApp() {
    geometryModeBtn.addEventListener('click', () => selectMode('geometry'));
    chemistryModeBtn.addEventListener('click', () => selectMode('chemistry'));
    resetBtn.addEventListener('click', resetApp);
    saveSessionBtn.addEventListener('click', handleSaveSession);
    loadSessionBtn.addEventListener('click', handleOpenLoadModal);
    closeModalBtn.addEventListener('click', () => loadModal.classList.add('hidden'));
    generateBtn.addEventListener('click', handleGenerateClick);
    imageInput.addEventListener('change', handleImageUpload);
    qaBtn.addEventListener('click', handleAskQuestion);
    qaInput.addEventListener('keydown', (e) => {
        if(e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleAskQuestion();
        }
    });

    initThree();
    checkSavedSessions();
}

initializeApp();