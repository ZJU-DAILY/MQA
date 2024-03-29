export type Encoder = {
    id: string;
    name: string;
    dimension: number;
    type: string[];
};

export const encoders: Encoder [] = [
    {id: '0', name: 'CLIP-Image', dimension: 512, type: ['Image']},
    {id: '1', name: 'CLIP-Text', dimension: 512, type: ['Text']},
    {id: '2', name: 'CLIP-Image+Text', dimension: 512, type: ['Image', 'Text']},
    {id: '3', name: 'CLIP4CIR-Image+Text', dimension: 640, type: ['Image', 'Text']},
];